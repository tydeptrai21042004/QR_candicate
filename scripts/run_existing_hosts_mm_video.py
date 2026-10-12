#!/usr/bin/env python3
"""Run MMWatermark source perturbations on existing six hosts + real encoded video.

No COCO/Imagenette host downloads. MMWatermark repository is cloned by
run_mmwatermark_official.py. The only optional media download is OpenCV vtest.avi
when the user does not provide a real video. Record partial/failed experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from qrwatermark.utils.watermark import prepare_binary_watermark
from qrwatermark.core.factory import build_method
from qrwatermark.evaluation.metrics import ber, nc, psnr, ssim

PROJECT = Path(__file__).resolve().parents[1]
VIDEO_SAMPLE = 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi'


def hash_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def validate_video(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size < 4096:
        raise ValueError(f'Video is absent or truncated: {path}')
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise ValueError(f'Video is not decodable: {path}')
        ok, frame=cap.read()
        if not ok or frame is None: raise ValueError(f'No frames in video: {path}')
        if (frame.shape[0]//4)*(frame.shape[1]//4)*4 < 4096:
            raise ValueError(f'Video is too small for 4096 watermark bits: {frame.shape}')
        return dict(path=str(path.resolve()),sha256=hash_file(path),first_frame_shape=list(frame.shape),bytes=path.stat().st_size)
    finally:
        cap.release()


def download_video(path: Path, url: str=VIDEO_SAMPLE, attempts: int=3) -> dict:
    """Download a real public video, atomic and bounded; never simulate frames."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        try:return validate_video(path)
        except ValueError:path.unlink()
    if url!=VIDEO_SAMPLE:raise ValueError('Only the pinned OpenCV sample URL is supported')
    error=None
    for attempt in range(attempts):
        tmp=path.with_name(path.stem+'.download'+path.suffix)
        try:
            tmp.unlink(missing_ok=True)
            req=urllib.request.Request(url,headers={'User-Agent':'GreenQuad-experimental-evaluation/1.0'})
            with urllib.request.urlopen(req,timeout=90) as r,tmp.open('wb') as dst:
                total=0
                while True:
                    chunk=r.read(1024*1024)
                    if not chunk:break
                    total+=len(chunk)
                    if total>200*1024*1024:raise ValueError('Video exceeds 200-MB safety limit')
                    dst.write(chunk)
            provenance=validate_video(tmp)
            tmp.replace(path)
            provenance.update(path=str(path.resolve()),source_url=url,sha256=hash_file(path))
            return provenance
        except Exception as e:
            error=e;tmp.unlink(missing_ok=True)
            print(f'Video download attempt {attempt+1}/{attempts}: {type(e).__name__}: {e}',flush=True)
            if attempt+1<attempts:time.sleep(1+attempt)
    raise RuntimeError(f'Could not download and decode real video: {error}')


def run_command(name: str, command: list[str], out: Path) -> dict:
    dest=out/'logs';dest.mkdir(parents=True,exist_ok=True)
    log=dest/(name+'.log')
    start=time.monotonic()
    try:
        with log.open('w',encoding='utf8') as f:
            f.write('COMMAND: '+' '.join(map(str,command))+'\n');f.flush()
            with subprocess.Popen(command,cwd=PROJECT,env={**os.environ,'PYTHONUNBUFFERED':'1'},
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                  errors='replace',bufsize=1) as p:
                for line in p.stdout:
                    print(f'[{name}] {line}',end='',flush=True)
                    f.write(line)
                code=p.wait()
        return dict(status='PASS' if code==0 else 'FAILED',returncode=code,
                    duration_seconds=round(time.monotonic()-start,2),log=str(log))
    except (OSError,ValueError) as exc:
        error=f'{type(exc).__name__}: {exc}'
        with log.open('a',encoding='utf8') as f:f.write('\nFAILED TO START: '+error+'\n')
        return dict(status='FAILED',error=error,duration_seconds=round(time.monotonic()-start,2),log=str(log))


def archive_reports(folder: Path) -> Path:
    z=folder/'MMWATERMARK_VIDEO_RESULTS.zip'
    with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as zipout:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p!=z and p.suffix.lower() not in {'.avi','.mp4','.mkv','.mov'}:
                zipout.write(p,p.relative_to(folder))
    return z


def clean_preflight(hosts: Path, out: Path, limit: int=6) -> dict:
    """Fail early on black-alpha watermark bugs, saving real preview images."""
    hosts=Path(hosts)
    paths=sorted(p for p in hosts.rglob('*') if p.is_file() and p.suffix.lower() in {'.bmp','.png','.jpg','.jpeg','.webp'})
    paths=paths[:limit]
    if len(paths)!=limit:
        raise ValueError(f'Expected {limit} usable existing hosts, found {len(paths)} in {hosts}')
    watermarks=[]; counts={};out.mkdir(parents=True,exist_ok=True)
    for i in (1,2):
        path=PROJECT/f'data/watermarks/watermark_{i}.png'
        wm=prepare_binary_watermark(path,size=64,background=255)
        black=int(np.count_nonzero(wm==0));white=int(np.count_nonzero(wm==255))
        if black==0 or white==0:raise AssertionError(f'Watermark {i} is constant after white-alpha compositing')
        watermarks.append(wm);counts[path.name]=dict(black=black,white=white)
        if not cv2.imwrite(str(out/f'expected_watermark_{i}.png'),wm):raise OSError('Cannot write preview')
    method=build_method('proposed');key=b'green-quad-local-clean-preflight'
    rows=[]
    for path in paths:
        host=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if host is None:raise ValueError(f'Cannot read existing host {path}')
        for i,wm in enumerate(watermarks,1):
            result=method.embed(host,wm,key=key)
            if result.side_info is not None:raise AssertionError('Blind embedding unexpectedly used side information')
            extracted=method.extract(result.image,key=key,watermark_shape=(64,64)).watermark
            b=float(ber(wm,extracted)); n=float(nc(wm,extracted))
            if b!=0 or not np.array_equal(extracted,wm):
                raise AssertionError(f'Clean extraction mismatch in {path.name} watermark {i}: BER={b}')
            if path==paths[0] and not cv2.imwrite(str(out/f'extracted_watermark_{i}.png'),extracted):
                raise OSError('Cannot write extracted watermark preview')
            rows.append(dict(host=path.name,watermark=i,ber=b,nc=n,psnr=float(psnr(host,result.image)),ssim=float(ssim(host,result.image))))
    manifest=dict(completed=True,host_count=len(paths),watermark_count=2,cases=len(rows),watermarks=counts,results=rows)
    (out/'clean_preflight.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return manifest


def run(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    hosts=Path(args.images_dir or PROJECT/'data/hosts/classical').resolve()
    provenance=dict(host_dir=str(hosts),dataset='existing_project_hosts',external_independent_hosts=False,
                    official_benchmark_protocol=False,video_source=None,method='green_quad_min_energy',
                    watermark_alpha='white background composite')
    status=dict(completed=False,benchmarks={},provenance=provenance)
    try:
        try:
            check=clean_preflight(hosts,out/'preflight',limit=args.min_images)
            status['benchmarks']['local_clean_preflight']=dict(status='PASS',cases=check['cases'],watermarks=check['watermarks'])
        except Exception as exc:
            status['benchmarks']['local_clean_preflight']=dict(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        if args.skip_mm:
            status['benchmarks']['mmwatermark']=dict(status='SKIPPED',reason='User specified --skip-mm')
        elif status['benchmarks']['local_clean_preflight']['status'] != 'PASS':
            status['benchmarks']['mmwatermark']=dict(status='BLOCKED',error='Clean watermark preflight failed; official evaluation not started')
        elif not (PROJECT/'scripts/run_mmwatermark_official.py').is_file():
            status['benchmarks']['mmwatermark']=dict(status='FAILED',error='MMWatermark runner is absent')
        else:
            mm_dir=out/'mmwatermark'
            mm_cmd=[sys.executable,'-u',str(PROJECT/'scripts/run_mmwatermark_official.py'),
                    '--images-dir',str(hosts),'--min-images',str(args.min_images),
                    '--max-images',str(args.max_images),'--official-repo',str(out/'official/MMWatermark-Robustness'),
                    '--output',str(mm_dir),'--severities',*map(str,args.severities),
                    '--bootstrap',str(args.bootstrap)]
            if not args.strict_mm:mm_cmd.append('--no-strict')
            status['benchmarks']['mmwatermark']=run_command('mmwatermark',mm_cmd,out)
            sp=mm_dir/'run_status.json'
            if sp.is_file():
                details=json.loads(sp.read_text())
                status['benchmarks']['mmwatermark']['official_perturbations_executed']=details.get('official_perturbations_executed',False)
                status['benchmarks']['mmwatermark']['coverage_complete']=details.get('coverage_complete',False)
                status['benchmarks']['mmwatermark']['official_commit']=details.get('official_repo_commit')
                status['benchmarks']['mmwatermark']['image_count']=details.get('images')
                if details.get('skipped') or details.get('failures'):
                    status['benchmarks']['mmwatermark']['status']='PARTIAL' if details.get('official_perturbations_executed') else 'FAILED'
        if args.skip_video:
            status['benchmarks']['video']=dict(status='SKIPPED',reason='User specified --skip-video')
        else:
            try:
                if args.video:
                    video=Path(args.video).resolve();provenance['video_source']=validate_video(video)
                    provenance['video_source']['origin']='user-supplied'
                else:
                    video=out/'inputs/opencv_vtest.avi';provenance['video_source']=download_video(video)
                    provenance['video_source']['origin']='OpenCV official public example'
                if not (PROJECT/'scripts/run_video_pipeline.py').is_file():
                    raise FileNotFoundError('Missing scripts/run_video_pipeline.py')
                for codec,ext in [('FFV1','.mkv'),('mp4v','.mp4')]:
                    for wm_idx in (1,2):
                        watermark=PROJECT/f'data/watermarks/watermark_{wm_idx}.png'
                        dest=out/'video'/codec/f'watermark_{wm_idx}'
                        name=f'video_{codec}_wm{wm_idx}'
                        cmd=[sys.executable,'-u',str(PROJECT/'scripts/run_video_pipeline.py'),
                             '--input-video',str(video),'--output-video',str(dest/f'watermarked{ext}'),
                             '--results-dir',str(dest),'--watermark-path',str(watermark),
                             '--codec',codec,'--max-frames',str(args.video_frames)]
                        status['benchmarks'][name]=run_command(name,cmd,out)
            except Exception as e:
                status['benchmarks']['video']=dict(status='FAILED',error=f'{type(e).__name__}: {e}')
        attempted=[v for v in status['benchmarks'].values() if v['status']!='SKIPPED']
        status['completed']=bool(attempted) and all(v['status']=='PASS' for v in attempted)
        return status
    finally:
        status_file=out/'execution_status.json'
        status_file.write_text(json.dumps(status,indent=2,allow_nan=False),encoding='utf8')
        archive_reports(out)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='GreenQuad existing-host MMWatermark + real encoded video')
    p.add_argument('--output',type=Path,default=Path('/kaggle/working/MMWATERMARK_VIDEO_RESULTS'))
    p.add_argument('--images-dir',type=Path,default=None,help='Default: existing six hosts in data/hosts/classical')
    p.add_argument('--min-images',type=int,default=6)
    p.add_argument('--max-images',type=int,default=6)
    p.add_argument('--severities',nargs='+',type=int,default=[1,2,3,4,5])
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--strict-mm',action='store_true',help='Fail if any original attack/severity is unavailable')
    p.add_argument('--video',type=Path,default=None,help='Use this video instead of downloading sample')
    p.add_argument('--video-frames',type=int,default=300)
    p.add_argument('--skip-mm',action='store_true')
    p.add_argument('--skip-video',action='store_true')
    args=p.parse_args(argv)
    if args.min_images<1 or args.max_images<args.min_images:p.error('Invalid host limits')
    if args.video_frames<1 or args.bootstrap<1:p.error('Invalid frame/bootstrap counts')
    return args


if __name__=='__main__':
    result=run(parse_args())
    print(json.dumps(result,indent=2))
    sys.exit(0 if result['completed'] else 1)
