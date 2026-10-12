#!/usr/bin/env python3
"""MMWatermark-Robustness image perturbation evaluation for Green-Quad.

Clones official source and runs ONLY discovered functions from its image_perturb.py.
Does not claim original paper's dataset/protocol or run substitution attacks.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, importlib.util, inspect, json, os, random, subprocess, sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from qrwatermark.core.factory import build_method
from qrwatermark.evaluation.metrics import ber, nc, psnr, ssim

OFFICIAL_URL = 'https://github.com/Jielin-Qiu/MMWatermark-Robustness.git'
IMAGE_EXT = {'.png','.bmp','.jpg','.jpeg','.tif','.tiff','.webp'}
# Only methods from the upstream module can be invoked. Never fall back to locally
# implemented attacks. Names are limited to common benchmark-corruption families.
KNOWN = ('gaussian_noise','shot_noise','impulse_noise','speckle_noise',
         'gaussian_blur','glass_blur','defocus_blur','motion_blur','zoom_blur',
         'fog','frost','snow','spatter','brightness','contrast','saturate',
         'pixelate','jpeg_compression','elastic_transform')


def prepare_repo(path: Path, *, clone: bool=True) -> str:
    path=Path(path)
    if not (path/'.git').exists():
        if not clone:
            raise RuntimeError(f'Official checkout absent: {path}')
        if path.exists() and any(path.iterdir()):
            raise RuntimeError('Refusing to clone into nonempty unverified directory')
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git','clone','--depth','1',OFFICIAL_URL,str(path)],check=True,timeout=180,
                       env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
    origin=subprocess.run(['git','-C',str(path),'remote','get-url','origin'],check=True,capture_output=True,text=True).stdout.strip()
    if origin.rstrip('/').removesuffix('.git').lower()!=OFFICIAL_URL.removesuffix('.git').lower():
        raise RuntimeError(f'Not the official repository: {origin}')
    dirty=subprocess.run(['git','-C',str(path),'status','--porcelain'],check=True,capture_output=True,text=True).stdout.strip()
    if dirty: raise RuntimeError('Official checkout contains uncommitted changes; cannot attest to source revision')
    sha=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if len(sha)!=40 or any(c not in '0123456789abcdef' for c in sha.lower()):
        raise RuntimeError('Invalid upstream revision')
    return sha


def find_module(repo: Path) -> Path:
    candidates = [p for p in (repo/'perturbations').rglob('image_perturb.py') if p.is_file()]
    if len(candidates)!=1:
        raise RuntimeError(f'Expected one official image_perturb.py under perturbations, found {len(candidates)}. '
                           'Inspect the upstream checkout; do not substitute local attacks.')
    return candidates[0]


def discover_names(module_path: Path) -> list[str]:
    # Parse before importing; --list-attacks works even when optional dependencies are absent.
    tree=ast.parse(Path(module_path).read_text(encoding='utf8'))
    names=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in KNOWN:
            names.append(node.name)
    return sorted(set(names))


def import_official(path: Path, repo: Path):
    # Import source from the verified checkout; intentionally no "mock attack" substitution.
    spec=importlib.util.spec_from_file_location('mmwatermark_upstream_image_perturb',path)
    if spec is None or spec.loader is None: raise RuntimeError('Cannot load official perturbation module')
    module=importlib.util.module_from_spec(spec)
    old=list(sys.path)
    old_bytecode=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        sys.path[:0]=[str(path.parent),str(repo/'perturbations'),str(repo)]
        spec.loader.exec_module(module)
    finally:
        sys.path[:]=old
        sys.dont_write_bytecode=old_bytecode
    return module


def call_official(fn, image_bgr: np.ndarray, severity: int, seed: int|None=None) -> np.ndarray:
    """Invoke a compatible upstream function; reject all unsupported signatures/results."""
    sig=inspect.signature(fn)
    params=list(sig.parameters.values())
    if not params or params[0].kind not in (inspect.Parameter.POSITIONAL_ONLY,inspect.Parameter.POSITIONAL_OR_KEYWORD):
        raise TypeError('Upstream perturbation requires an unsupported first parameter')
    if len(params)<2 or params[1].name not in {'severity','level'} or params[1].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,inspect.Parameter.POSITIONAL_OR_KEYWORD):
        raise TypeError('Upstream perturbation does not expose (image, severity/level)')
    if any(p.default is inspect.Parameter.empty for p in params[2:] if p.kind not in
           (inspect.Parameter.VAR_POSITIONAL,inspect.Parameter.VAR_KEYWORD)):
        raise TypeError('Upstream perturbation requires additional unsupported arguments')
    rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Conventional image-corruption implementations accept PIL RGB images.
    # Restore global RNG state after calling an upstream stochastic perturbation.
    # Identical host/watermark/attack/severity => reproducible independent trial.
    if seed is None:
        result=fn(Image.fromarray(rgb),severity)
    else:
        np_state=np.random.get_state();py_state=random.getstate()
        try:
            np.random.seed(int(seed)%2**32);random.seed(int(seed))
            result=fn(Image.fromarray(rgb),severity)
        finally:
            np.random.set_state(np_state);random.setstate(py_state)
    if isinstance(result,Image.Image): result=np.asarray(result.convert('RGB'))
    else: result=np.asarray(result)
    if result.shape!=rgb.shape:
        raise ValueError(f'Perturbation changed geometry: {result.shape} != {rgb.shape}')
    if not np.issubdtype(result.dtype,np.number) or not np.all(np.isfinite(result)):
        raise ValueError('Non-numeric or non-finite official perturbation output')
    if result.dtype!=np.uint8:
        if result.size and float(np.min(result))>=0 and float(np.max(result))<=1.0 and result.dtype.kind=='f':
            result=result*255
        result=np.clip(np.rint(result),0,255).astype(np.uint8)
    return cv2.cvtColor(result,cv2.COLOR_RGB2BGR)


def read_images(folder: Path,limit: int, *,min_size: int=128) -> list[tuple[str,np.ndarray]]:
    if limit<1: raise ValueError('max-images must be positive')
    if not folder.is_dir(): raise FileNotFoundError(f'Independent image directory missing: {folder}')
    all_paths=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    out=[]; names=set()
    for p in all_paths:
        im=cv2.imread(str(p),cv2.IMREAD_COLOR)
        if im is None or min(im.shape[:2])<min_size: continue
        # Files in different folders may share a stem; use a stable path-derived id.
        rel=str(p.relative_to(folder)); name=hashlib.sha256(rel.encode()).hexdigest()[:14]
        if name in names: raise AssertionError('image ID collision')
        names.add(name);out.append((name,im))
        if len(out)>=limit:break
    if not out:raise RuntimeError('No eligible images; supply >=128px 3-channel images')
    return out


def score_rows(rows: list[dict], seed: int=3101, n_boot: int=500) -> list[dict]:
    """Cluster at the host level, not at attack/seed/watermark level."""
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        groups[(r['attack'],r['severity'])][r['host_id']].append(r)
    rng=np.random.default_rng(seed); summary=[]
    for (attack,sev), hosts in sorted(groups.items()):
        hids=sorted(hosts)
        means=np.array([[np.mean([z[k] for z in hosts[h]]) for k in ('nc','ber')] for h in hids])
        index=rng.integers(0,len(hids),size=(n_boot,len(hids)))
        samples=means[index].mean(axis=1)
        avg=means.mean(axis=0)
        lo=np.percentile(samples,2.5,axis=0); hi=np.percentile(samples,97.5,axis=0)
        summary.append(dict(attack=attack,severity=sev,hosts=len(hids),observations=sum(map(len,hosts.values())),
                            mean_nc=float(avg[0]),nc_ci_low=float(lo[0]),nc_ci_high=float(hi[0]),
                            mean_ber=float(avg[1]),ber_ci_low=float(lo[1]),ber_ci_high=float(hi[1])))
    return summary


def write_csv(path: Path,records: list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not records: return
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def trial_seed(global_seed:int, host_id:str, wm_id:int, attack:str, severity:int) -> int:
    raw=f'{global_seed}|{host_id}|{wm_id}|{attack}|{severity}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4],'big')

def run(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    status={'repository':OFFICIAL_URL,'official_repo_commit':None,'method':'green_quad_min_energy',
            'period':66,'payload':4096,'official_perturbations_executed':False,'skipped':{},
            'failures':[],'completed':False}
    try:
        repo=Path(args.official_repo)
        sha=prepare_repo(repo,clone=not args.no_clone)
        status['official_repo_commit']=sha
        script=find_module(repo);status['official_module']=str(script.relative_to(repo))
        status['official_module_sha256']=hashlib.sha256(script.read_bytes()).hexdigest()
        available=discover_names(script);status['discovered_attacks']=available
        if args.list_attacks:
            print(json.dumps({'official_commit':sha,'available':available},indent=2));status['completed']=True;return status
        requested=([x.strip() for x in args.attacks.split(',') if x.strip()] if args.attacks else available)
        missing=set(requested)-set(available)
        if missing:raise RuntimeError(f'Official functions not found: {sorted(missing)}; available={available}')
        if not requested:raise RuntimeError('No compatible named official perturbation functions discovered; fail closed')
        if len(args.severities)!=len(set(args.severities)) or any(s<1 or s>5 for s in args.severities):
            raise ValueError('severities must be unique integers from 1 to 5')
        official=import_official(script,repo)
        attacks={name:getattr(official,name) for name in requested}
        images=read_images(Path(args.images_dir),args.max_images)
        if len(images)<args.min_images:raise RuntimeError(f'Only {len(images)} eligible independent images; require {args.min_images}')
        project_root=Path(__file__).resolve().parents[1]
        wm_paths=[project_root/'data/watermarks/watermark_1.png',project_root/'data/watermarks/watermark_2.png']
        wms=[cv2.imread(str(path),cv2.IMREAD_GRAYSCALE) for path in wm_paths]
        if any(w is None or w.shape!=(64,64) for w in wms):raise RuntimeError('Expected two actual 64x64 watermark images in repository')
        method=build_method('proposed');key=args.key.encode('utf8')
        trials=[];ok=set();skipped={}
        preflight_image=images[0][1]
        # Strictly preflight every attack/severity combination, so partial results aren't
        # silently reported as a full official benchmark.
        for attack,fn in attacks.items():
            for severity in args.severities:
                label=f'{attack}:{severity}'
                try:
                    altered=call_official(fn,preflight_image,severity,seed=trial_seed(args.seed,'preflight',0,attack,severity))
                    if altered.shape!=preflight_image.shape:raise RuntimeError('geometry mismatch')
                    ok.add((attack,severity))
                except Exception as e:
                    skipped[label]=f'{type(e).__name__}: {e}'
        status['skipped']=skipped
        if skipped and args.strict:
            raise RuntimeError(f'Official attack preflight failed for {len(skipped)} combinations; see run_status.json')
        if not ok:raise RuntimeError('No official perturbation executed successfully')
        for host_id,im in images:
            for wm_id,wm in enumerate(wms,1):
                result=method.embed(im,wm,key=key)
                if result.side_info is not None:raise AssertionError('Proposal not blind')
                clean=method.extract(result.image,key=key,watermark_shape=(64,64)).watermark
                clean_ber=ber(wm,clean)
                if clean_ber!=0:raise AssertionError(f'Clean BER not zero for host {host_id}; wm {wm_id}')
                trials.append(dict(host_id=host_id,watermark=wm_id,attack='clean',severity=0,
                                   psnr=float(psnr(im,result.image)),ssim=float(ssim(im,result.image)),
                                   nc=float(nc(wm,clean)),ber=clean_ber))
                for attack,severity in sorted(ok):
                    try:
                        attack_seed=trial_seed(args.seed,host_id,wm_id,attack,severity)
                        attacked=call_official(attacks[attack],result.image,severity,seed=attack_seed)
                        rec=method.extract(attacked,key=key,watermark_shape=(64,64)).watermark
                        trials.append(dict(host_id=host_id,watermark=wm_id,attack=attack,severity=severity,
                                           psnr=float(psnr(im,result.image)),ssim=float(ssim(im,result.image)),
                                           nc=float(nc(wm,rec)),ber=float(ber(wm,rec))))
                    except Exception as e:
                        status['failures'].append(dict(host_id=host_id,watermark=wm_id,attack=attack,
                                                       severity=severity,error=f'{type(e).__name__}: {e}'))
                        if args.strict:raise
        status['images']=len(images);status['watermarks']=len(wms)
        status['expected_rows']=len(images)*len(wms)*(1+len(ok))
        status['actual_rows']=len(trials)
        if len(trials)!=status['expected_rows']:
            raise RuntimeError(f'Incomplete benchmark rows: {len(trials)}/{status["expected_rows"]}')
        status['official_perturbations_executed']=True
        status['executed_attack_severity_pairs']=[f'{name}:{severity}' for name,severity in sorted(ok)]
        write_csv(dest/'per_sample.csv',trials)
        write_csv(dest/'attack_summary.csv',score_rows(trials,seed=args.seed,n_boot=args.bootstrap))
        status['completed']=True
        return status
    finally:
        (dest/'run_status.json').write_text(json.dumps(status,indent=2,allow_nan=False),encoding='utf8')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Clone/run original MMWatermark image perturbations against Green-Quad')
    p.add_argument('--images-dir',type=Path,default=None,help='Independent host images; no default to development hosts')
    p.add_argument('--official-repo',type=Path,default=Path('third_party/MMWatermark-Robustness'))
    p.add_argument('--no-clone',action='store_true')
    p.add_argument('--list-attacks',action='store_true')
    p.add_argument('--output',type=Path,default=Path('results/mmwatermark'))
    p.add_argument('--max-images',type=int,default=100)
    p.add_argument('--min-images',type=int,default=1)
    p.add_argument('--severities',nargs='+',type=int,default=[1,2,3,4,5])
    p.add_argument('--attacks',default='',help='Comma-separated names from --list-attacks; default all supported')
    p.add_argument('--strict',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--seed',type=int,default=3101)
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--key',default='green-quad-mmwatermark-benchmark-v1')
    a=p.parse_args(argv)
    if not a.list_attacks and a.images_dir is None:p.error('--images-dir is required for evaluation')
    if a.bootstrap<1:p.error('bootstrap must be positive')
    return a

if __name__=='__main__':
    a=parse_args()
    try:
        result=run(a)
        print(json.dumps(result,indent=2));sys.exit(0 if result['completed'] else 1)
    except Exception as e:
        print(f'FAILED: {type(e).__name__}: {e}',file=sys.stderr)
        raise
