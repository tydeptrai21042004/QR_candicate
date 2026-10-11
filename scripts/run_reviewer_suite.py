"""Reproducible reviewer experiments for the UNCHANGED Green-Quad T=66 method.

Usage: PYTHONPATH=src python scripts/run_reviewer_suite.py --mode smoke --sections all
Full mode uses the existing 6-host corpus by default; add new independent
images via --hosts-dir for meaningful external generalization evidence.

Never adds pilots, side information, ECC or model inference to Green-Quad.
"""
from __future__ import annotations
import argparse, csv, json, hashlib, platform, sys, time, traceback
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from _common import resolve,load_yaml
from qrwatermark.proposed.method import GreenQuadWatermark
from qrwatermark.core.factory import build_method
from qrwatermark.core.config import ProposedConfig
from qrwatermark.proposed.green_quad_min_energy import _lattice_centres, optimal_integer_projection,greedy_integer_projection
from qrwatermark.proposed.block_carrier import keyed_blocks,_gather_blocks
from qrwatermark.proposed.green_quad_min_energy import _to_cells
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.watermark import prepare_binary_watermark,scrambled_bits_from_watermark
from qrwatermark.evaluation.metrics import psnr,ssim,ber,nc
from qrwatermark.evaluation.benchmark import run_benchmark
from qrwatermark.evaluation.fairness import tune_baseline_to_psnr
from qrwatermark.evaluation.review_statistics import statistical_report,roc_points,calibrate_threshold
from qrwatermark.evaluation.review_protocol import split_hosts,manifest
from qrwatermark.attacks import apply_attack

ROOT=Path(__file__).resolve().parents[1]
WM_DEFAULT=ROOT/'data/watermarks'

def dump_csv(path, rows, columns=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows,columns=columns if not len(rows) else None)
    df.to_csv(path,index=False)
    return df

def get_wm(path,size=64):
    return prepare_binary_watermark(path,size)

def proposal(size=64,period=66):
    return GreenQuadWatermark(ProposedConfig(watermark_size=size,integer_conv4_period=period))

def keyed_digest(key):
    return hashlib.sha256(key).hexdigest()[:16]

def score(expected,got):
    # Correct/wrong ownership verification requires expected watermark.
    return 1.0-ber(expected,got)

def attack_cases(path,quick=False):
    cases=[]
    for entry in load_yaml(path)['attacks']:
        for seed in entry.get('seeds',[None]):
            cases.append((entry['name'],dict(entry.get('params',{})),seed))
    if quick:
        names={'clean','jpeg','rotation_unregistered','subpixel_translation','green_carrier_null'}
        cases=[c for c in cases if c[0] in names and (c[0]!='jpeg' or c[1]['quality']==40)
              and (c[0]!='rotation_unregistered' or c[1]['angle']==2.5)
              and (c[0]!='green_carrier_null' or c[1]['strength']==1.0)]
    return cases

def attack_evaluation(split,wms,key,out,attack_config,quick):
    method=proposal()
    rows=[]
    for hp in split['test'][:1 if quick else None]:
        host=read_color(hp)
        for wp in wms[:1 if quick else None]:
            wm=get_wm(wp); st=method.embed(host,wm,key=key).image
            for name,kw,seed in attack_cases(attack_config,quick):
                opts=kw.copy()
                if seed is not None:opts['seed']=seed
                corrupted=apply_attack(name,st,**opts)
                ext=method.extract(corrupted,key=key,watermark_shape=wm.shape).watermark
                rows.append(dict(method=method.name,host=hp.stem,watermark=wp.stem,attack=name,
                                attack_params=json.dumps(kw,sort_keys=True),seed=seed,
                                embedding_psnr=psnr(host,st),embedding_ssim=ssim(host,st),
                                attacked_psnr=psnr(host,corrupted),ber=ber(wm,ext),nc=nc(wm,ext)))
        dump_csv(out/'attacks/raw.csv',rows)
    df=dump_csv(out/'attacks/raw.csv',rows)
    statistical_report(df,out/'attacks',n_boot=200 if quick else 1000)


def security(split,wms,key,out,quick):
    method=proposal(); rows=[]
    # Explicit calibration/test split: never derive threshold from test negatives.
    for split_name in ('validation','test'):
        for hp in split[split_name][:1 if quick else None]:
            host=read_color(hp)
            for wp in wms[:1 if quick else None]:
                wm=get_wm(wp); stego=method.embed(host,wm,key=key).image
                cases=[('positive_correct_key',stego,key,wm,1),
                       ('negative_unmarked',host,key,wm,0)]
                flipped=bytearray(key);flipped[0]^=1
                cases.append(('negative_one_bit_key_flip',stego,bytes(flipped),wm,0))
                for index in range(4 if quick else 32):
                    # Derive adversarial keys reproducibly from a domain-separated seed.
                    wrong=hashlib.sha256(b'wrong-key-audit-v1'+index.to_bytes(4,'big')+key).digest()
                    cases.append(('negative_wrong_key',stego,wrong,wm,0))
                # Both genuine and negative templates are deterministic and separate.
                other=get_wm(wms[-1]) if len(wms)>1 and wms[-1]!=wp else (255-wm)
                cases.append(('negative_unrelated_template',stego,key,other,0))
                for label,img,k,template,truth in cases:
                    ext=method.extract(img,key=k,watermark_shape=wm.shape).watermark
                    rows.append(dict(split=split_name,host=hp.stem,watermark=wp.stem,
                                     case=label,truth=truth,score=score(template,ext),
                                     extraction_ber=ber(template,ext),key_tag=keyed_digest(k)))
        dump_csv(out/'security/raw.csv',rows)
    df=dump_csv(out/'security/raw.csv',rows)
    val=df[df.split=='validation'];test=df[df.split=='test']
    val_neg=val[val.truth==0].score.to_numpy()
    threshold=calibrate_threshold(val_neg,0.01)
    positive=test[test.truth==1].score.to_numpy()
    negative=test[test.truth==0].score.to_numpy()
    roc,auc=roc_points(positive,negative)
    roc.to_csv(out/'security/roc.csv',index=False)
    report=dict(score='1-BER against the claimed reference',
                threshold_calibration='validation negatives only',target_fpr=0.01,
                threshold=threshold,heldout_auc=auc,
                heldout_tpr=float((positive>=threshold).mean()),
                heldout_fpr=float((negative>=threshold).mean()),
                validation_negative_count=len(val_neg),test_positive_count=len(positive),
                test_negative_count=len(negative),
                caution='Null samples sharing a host are dependent; empirical FPR precision is limited by host count.')
    (out/'security/report.json').write_text(json.dumps(report,indent=2))


def energy_study(split,wms,key,out,quick):
    rows=[];m=proposal()
    for hp in split['test'][:1 if quick else None]:
        host=read_color(hp)
        for wp in wms[:1 if quick else None]:
            wm=get_wm(wp)
            rr,cc=keyed_blocks(host.shape[:2],wm.size//4,key)
            values=_to_cells(_gather_blocks(host[:,:,1],rr,cc)).reshape(-1,4)
            bit=scrambled_bits_from_watermark(wm,0)
            sums=values.astype(np.int32).sum(axis=1)
            base=_lattice_centres(sums,bit,66)
            stats={};fixed=optimal_integer_projection(values,base,stats_out=stats)
            # Two immediately adjacent feasible lattice targets: diagnostic only.
            prev=base-66;following=base+66
            candidate_cost=((fixed.astype(np.int32)-values.astype(np.int32))**2).sum(axis=1)
            better=np.zeros(len(base),dtype=bool)
            best=candidate_cost.copy()
            for t in (prev,following):
                valid=(t>=0)&(t<=1020)
                if valid.any():
                    repl=optimal_integer_projection(values[valid],t[valid])
                    costs=((repl.astype(np.int32)-values[valid].astype(np.int32))**2).sum(axis=1)
                    idx=np.flatnonzero(valid)
                    improvement=costs<best[idx]
                    better[idx[improvement]]=True
                    best[idx]=np.minimum(best[idx],costs)
            greedy=greedy_integer_projection(values,base)
            assert np.all(fixed.astype(np.int32).sum(axis=1)==base)
            assert np.all(((fixed.astype(np.int32)-values.astype(np.int32))**2).sum(axis=1)
                          <= ((greedy.astype(np.int32)-values.astype(np.int32))**2).sum(axis=1))
            embed=m.embed(host,wm,key=key)
            recovered=m.extract(embed.image,key=key,watermark_shape=wm.shape).watermark
            rows.append(dict(host=hp.stem,watermark=wp.stem,carrier_count=len(base),
                             fixed_centre_energy=int(candidate_cost.sum()),
                             adjacent_centre_energy_lower_bound=int(best.sum()),
                             adjacent_better_carriers=int(better.sum()),
                             fixed_vs_greedy_energy_gain=int(((greedy.astype(np.int32)-values.astype(np.int32))**2).sum()-candidate_cost.sum()),
                             projection_fallback_cells=stats.get('fallback_cells',0),
                             clipped_pixels_in_input=int(np.count_nonzero((values==0)|(values==255))),
                             changed_pixels_at_boundary=int(np.count_nonzero((fixed==0)|(fixed==255))),
                             clean_ber=ber(wm,recovered),embedding_psnr=psnr(host,embed.image)))
    for constant in (0,1,127,254,255):
        host=np.full((512,512,3),constant,dtype=np.uint8)
        wm=get_wm(wms[0]);res=m.embed(host,wm,key=key)
        got=m.extract(res.image,key=key,watermark_shape=wm.shape).watermark
        rows.append(dict(host=f'constant_{constant}',watermark=wms[0].stem,carrier_count=wm.size,
                         projection_fallback_cells=res.metadata.get('fallback_cells'),
                         clipped_pixels_in_input=wm.size*4*int(constant in (0,255)),
                         clean_ber=ber(wm,got),embedding_psnr=psnr(host,res.image)))
    dump_csv(out/'optimality/integer_projection.csv',rows)


def capacity(split,wms,key,out,quick):
    m=proposal();rows=[]
    sizes=(16,32,64,128,256) if not quick else (16,64,128)
    periods=(48,58,66,74,82) if not quick else (66,)
    for hp in split['test'][:1 if quick else None]:
        host=read_color(hp)
        for size in sizes:
            if (host.shape[0]//4)*(host.shape[1]//4)*4 < size*size:continue
            wm=get_wm(wms[0],size)
            for period in periods:
                method=proposal(size=size,period=period)
                t0=time.perf_counter();emb=method.embed(host,wm,key=key);te=time.perf_counter()-t0
                t0=time.perf_counter();decoded=method.extract(emb.image,key=key,watermark_shape=wm.shape).watermark
                tx=time.perf_counter()-t0
                assert ber(wm,decoded)==0, 'Clean extraction failed during capacity sweep'
                for name,kw,seed in [('clean',{},None),('jpeg',{'quality':40},None),
                                     ('gaussian_noise',{'variance':0.01},3101)]:
                    if seed is not None:kw=dict(kw,seed=seed)
                    attacked=apply_attack(name,emb.image,**kw)
                    result=method.extract(attacked,key=key,watermark_shape=wm.shape).watermark
                    rows.append(dict(host=hp.stem,payload_bits=size*size,bpp=size*size/(host.shape[0]*host.shape[1]),
                                     watermark_size=size,period=period,attack=name,
                                     psnr=psnr(host,emb.image),ssim=ssim(host,emb.image),
                                     ber=ber(wm,result),nc=nc(wm,result),
                                     embed_ms=1000*te,extract_ms=1000*tx,
                                     single_run_fps=1/(te+tx)))
        dump_csv(out/'capacity/capacity.csv',rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        df=pd.DataFrame(rows)
        for metric,ylabel,filename in [('ber','BER','distortion_robustness.png'),
                                      ('psnr','PSNR (dB)','payload_distortion.png')]:
            fig,ax=plt.subplots(figsize=(7,4))
            subset=df[df.attack.eq('jpeg') if metric=='ber' else df.attack.eq('clean')]
            for period,ss in subset.groupby('period'):
                aa=ss.groupby('payload_bits')[metric].mean().sort_index()
                ax.plot(aa.index,aa.values,marker='o',label=f'T={period}')
            ax.set_xscale('log',base=2);ax.set_xlabel('Payload bits');ax.set_ylabel(ylabel)
            ax.legend();fig.tight_layout()
            fig.savefig(out/'capacity'/filename,dpi=160);plt.close(fig)
    except ImportError:
        # CSV always remains authoritative, even without matplotlib installed.
        pass


def runtime(split,wms,key,out,quick):
    method_names=['proposed','direct_pixel_qim_control','su2014_qr','su2016_hessenberg',
                  'su2020_schur','chen2021_qqrd','nha2022_improved_qr']
    rows=[];n=2 if quick else 100;warm=1 if quick else 12
    for name in method_names:
        config='configs/methods/direct_qim_control.yaml' if name=='direct_pixel_qim_control' else f'configs/methods/{name}.yaml'
        method=build_method(name,resolve(config))
        for hp in split['test'][:1 if quick else None]:
            host=read_color(hp);wm=get_wm(wms[0]);times=[]
            try:
                for r in range(warm+n):
                    t0=time.perf_counter_ns();emb=method.embed(host,wm,key=key)
                    t1=time.perf_counter_ns();ext=method.extract(emb.image,key=key,side_info=emb.side_info,watermark_shape=wm.shape)
                    t2=time.perf_counter_ns()
                    if r>=warm:times.append(((t1-t0)*1e-6,(t2-t1)*1e-6))
                arr=np.asarray(times);total=arr.sum(axis=1)
                rows.append(dict(method=method.name,host=hp.stem,repetitions=n,warmups=warm,
                                 embed_median_ms=float(np.median(arr[:,0])),extract_median_ms=float(np.median(arr[:,1])),
                                 combined_median_ms=float(np.median(total)),
                                 combined_p95_ms=float(np.percentile(total,95)),
                                 combined_p99_ms=float(np.percentile(total,99)),
                                 combined_fps=1000/np.median(total),
                                 clean_ber=ber(wm,ext.watermark),
                                 timing_note='Warm process, no attack or PSNR calibration in throughput'))
            except Exception as e:
                rows.append(dict(method=name,host=hp.stem,error=repr(e)))
            dump_csv(out/'runtime/runtime.csv',rows)
    (out/'runtime/environment.json').write_text(json.dumps({
        'python':sys.version,'platform':platform.platform(),'cpu_count':__import__('os').cpu_count(),
        'numpy':np.__version__,'opencv':cv2.__version__,
        'cv2_threads':cv2.getNumThreads(),'timing_clock':'perf_counter_ns',
        'note':'Runtime not cross-platform comparable; per-host latency and FPS, no codec/attack time.'},indent=2))


def baseline_comparison(split,wms,key,out,attack_path,quick,psnr_tolerance=0.2):
    cfg=load_yaml('configs/experiments/reviewer_matched.yaml')
    names=cfg['methods']
    if quick:
        # Fast smoke mode runs two paper controls; FULL mode runs all five.
        names=[spec for spec in names if spec['name'] in
               ('proposed','direct_pixel_qim_control','su2014_qr','nha2022_improved_qr')]
    reference=proposal()
    # Calibrate ONLY on validation hosts; freeze each baseline before touching test.
    diagnostics=[]; methods=[reference]
    for spec in names[1:]:
        name=spec['name'];original=build_method(name,resolve(spec['config']))
        scales=[];failures=[]
        for hp in split['validation']:
            host=read_color(hp)
            for wp in wms[:1 if quick else None]:
                wm=get_wm(wp)
                try:
                    ref=reference.embed(host,wm,key=key)
                    _,_,diag=tune_baseline_to_psnr(original,host,wm,key=key,
                                                  target_psnr_db=psnr(host,ref.image),
                                                  scales=([0.15,0.35,0.6,1.0,1.5] if quick else None))
                    if 'strength_scale' in diag:scales.append(float(diag['strength_scale']))
                except Exception as ex:
                    failures.append(f'{hp.name}/{wp.name}: {ex}')
        fields=getattr(original,'strength_fields',None)
        if not fields:
            field=getattr(original,'strength_field',None)
            fields=(field,) if field else ('quant_step',)
        if not scales:
            diagnostics.append(dict(method=name,error='No successful validation calibration',failure_details=str(failures)))
            continue
        scale=float(np.median(scales));new={fld:float(getattr(original.config,fld))*scale for fld in fields}
        tuned=original.__class__(replace(original.config,**new));methods.append(tuned)
        diagnostics.append(dict(method=name,selected_scale=scale,strength_fields=','.join(fields),
                                validation_pairs=len(scales),failure_details=str(failures),
                                calibration='validation-only global strength; frozen on test'))
    dump_csv(out/'baselines/calibration.csv',diagnostics)
    attacks=load_yaml(attack_path)['attacks']
    if quick:
        attacks=[e for e in attacks if e['name']=='clean' or
                 (e['name']=='jpeg' and e.get('params',{}).get('quality')==40)]
    data=run_benchmark(methods,split['test'][:1 if quick else None],wms[:1 if quick else None],
                       attacks,key=key,run_dir=out/'baselines',save_images=False,
                       match_psnr_to_proposed=False,watermark_size=64)
    clean=data[data.attack=='clean'][['method','host','watermark','embedding_psnr']].copy()
    proposal_psnr=clean[clean.method=='green_quad_min_energy'][['host','watermark','embedding_psnr']].rename(columns={'embedding_psnr':'target_psnr'})
    fair=clean.merge(proposal_psnr,on=['host','watermark'],validate='many_to_one')
    fair['psnr_gap_db']=fair.embedding_psnr-fair.target_psnr
    fair['within_tolerance']=fair.psnr_gap_db.abs()<=psnr_tolerance
    fair['payload_bits']=4096
    fair['method_note']=np.where(fair.method=='direct_pixel_qim_control','New non-paper scalar QIM control','Registered paper or proposed method')
    dump_csv(out/'baselines/psnr_audit.csv',fair)
    # Metrics for poorly-matched baselines are supplied but NOT mixed into fair wins.
    eligible=fair[['method','host','watermark','within_tolerance']]
    data=data.merge(eligible,on=['method','host','watermark'],validate='many_to_one')
    data.to_csv(out/'baselines/metrics_with_eligibility.csv',index=False)
    statistical_report(data,out/'baselines',n_boot=200 if quick else 1000)
    comparable=data[data.within_tolerance]
    if len(comparable):statistical_report(comparable,out/'baselines/matched_only',n_boot=200 if quick else 1000)
    # SECONDARY OPERATING-POINT study: tune ONLY the host PSNR (never attack BER)
    # separately for each evaluation host. It is not the locked-validation result.
    if not quick:
        original=[build_method(spec['name'],resolve(spec['config'])) for spec in names]
        calibrated=run_benchmark(original,split['test'],wms,attacks,key=key,
                                 run_dir=out/'baselines/operating_point_psnr_calibrated',
                                 save_images=False,match_psnr_to_proposed=True,watermark_size=64)
        calibrated['within_tolerance']=calibrated['matched_psnr_error_db'].abs()<=psnr_tolerance
        calibrated.to_csv(out/'baselines/operating_point_psnr_calibrated'/'metrics_eligibility.csv',index=False)
        statistical_report(calibrated[calibrated.within_tolerance],
                           out/'baselines/operating_point_psnr_calibrated'/'matched_only',n_boot=1000)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','full'],default='smoke')
    p.add_argument('--sections',nargs='+',default=['all'],
                   choices=['all','baselines','security','attacks','optimality','capacity','runtime'])
    p.add_argument('--hosts-dir',type=Path,default=resolve('data/hosts/classical'))
    p.add_argument('--watermarks-dir',type=Path,default=WM_DEFAULT)
    p.add_argument('--output',type=Path,default=Path('results/reviewer_suite'))
    p.add_argument('--attacks-config',default='configs/attacks/reviewer_stress.yaml')
    p.add_argument('--key',default='review-audit-2026-fixed-key')
    p.add_argument('--split-seed',type=int,default=3101)
    p.add_argument('--psnr-tolerance',type=float,default=0.2)
    p.add_argument('--fail-fast',action='store_true')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    hosts=sorted(a.hosts_dir.glob('*.bmp'))+sorted(a.hosts_dir.glob('*.png'))
    wms=sorted(a.watermarks_dir.glob('*.png'))
    if not wms:raise ValueError('No watermark PNG files')
    split=split_hosts(hosts,a.split_seed)
    manifest(split,wms,a.attacks_config,out,a.split_seed)
    key=a.key.encode();quick=(a.mode=='smoke')
    sections=['security','attacks','optimality','capacity','runtime','baselines'] if 'all' in a.sections else a.sections
    failed={}
    actions={
        'security':lambda:security(split,wms,key,out,quick),
        'attacks':lambda:attack_evaluation(split,wms,key,out,a.attacks_config,quick),
        'optimality':lambda:energy_study(split,wms,key,out,quick),
        'capacity':lambda:capacity(split,wms,key,out,quick),
        'runtime':lambda:runtime(split,wms,key,out,quick),
        'baselines':lambda:baseline_comparison(split,wms,key,out,a.attacks_config,quick,a.psnr_tolerance)}
    for section in sections:
        print(f'Starting {section}',flush=True)
        try:actions[section]()
        except Exception as ex:
            failed[section]=traceback.format_exc()
            (out/f'{section}_ERROR.txt').write_text(failed[section])
            print(f'FAILED: {section}: {ex}',flush=True)
            if a.fail_fast:raise
        else:print(f'Completed {section}',flush=True)
    (out/'run_status.json').write_text(json.dumps({'mode':a.mode,'completed':[k for k in sections if k not in failed],
                                                 'failed':list(failed),'proposed_unchanged':True},indent=2))
    if failed:raise SystemExit(f'{len(failed)} sections failed; see *_ERROR.txt in {out}')
    print('Results:',out)
if __name__=='__main__':main()
