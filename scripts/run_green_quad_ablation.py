"""Reproducible single-proposal ablation and performance evaluation.

Runs exactly one proposed algorithm. The controls vary its channel, period,
optional permutation, and integer projection, not other proposal families.
Each row is a full host/watermark/attack result; no silently omitted attacks.
Measured FPS is specific to the machine executing the script.
"""
from __future__ import annotations
import argparse,ast,csv,json,platform,sys,time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qrwatermark.core.config import ProposedConfig
from qrwatermark.proposed.method import GreenQuadWatermark
from qrwatermark.utils.watermark import prepare_binary_watermark
from qrwatermark.attacks import apply_attack
from qrwatermark.evaluation.metrics import psnr,ssim,ber,nc

HOSTS=('airplane','girl','lenna','manhattan','pepper','safari')
WMS=('watermark_1','watermark_2')
KEYS=(b'reviewer-speed-recheck-2026',b'independent-key-B',b'independent-key-C',b'independent-key-D',b'independent-key-E')
# Full algorithm is ONE family: green T66 is production; all other entries are internal ablations.
VARIANTS={
 'FINAL_GREEN_T66':{},
 'period_64':{'integer_conv4_period':64},
 'period_68':{'integer_conv4_period':68},
 'arnold_10':{'arnold_iterations':10},
 'channel_blue':{'channel':0},
 'channel_red':{'channel':2},
 'nonoptimal_greedy_control':{'projection':'greedy_control'},
}

def write_csv(path,rows,header):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=header);writer.writeheader();writer.writerows(rows)

def variant_method(name):
    cfg=ProposedConfig(**{'ablation_mode':name!='FINAL_GREEN_T66',**VARIANTS[name]})
    return GreenQuadWatermark(cfg)

def read_dataset(mode):
    hosts=HOSTS if mode=='full' else ('girl',)
    watermarks=WMS if mode=='full' else ('watermark_1',)
    data=[]
    for h in hosts:
        host=cv2.imread(str(ROOT/'data'/'hosts'/'classical'/f'{h}.bmp'),cv2.IMREAD_COLOR)
        if host is None:raise FileNotFoundError(h)
        for w in watermarks:
            wm=prepare_binary_watermark(ROOT/'data'/'watermarks'/f'{w}.png')
            data.append((h,w,host,wm))
    return data

def get_attacks():
    with (ROOT/'configs'/'attacks'/'green_quad_final_25.csv').open(newline='') as f:
        return [(a['attack'],ast.literal_eval(a['parameters'])) for a in csv.DictReader(f)]

def speed_test(method,host,wm,key,repeats):
    for _ in range(8):
        stego=method.embed(host,wm,key=key).image
        method.extract(stego,key=key,watermark_shape=wm.shape)
    emb=[];ext=[]
    for _ in range(repeats):
        t0=time.perf_counter_ns();stego=method.embed(host,wm,key=key).image;t1=time.perf_counter_ns()
        decoded=method.extract(stego,key=key,watermark_shape=wm.shape).watermark;t2=time.perf_counter_ns()
        if not np.array_equal(decoded,wm):raise AssertionError('clean roundtrip failed')
        emb.append((t1-t0)/1e6);ext.append((t2-t1)/1e6)
    combo=np.asarray(emb)+np.asarray(ext)
    return dict(embed_median_ms=float(np.median(emb)),extract_median_ms=float(np.median(ext)),combined_median_ms=float(np.median(combo)),combined_p95_ms=float(np.percentile(combo,95)),combined_fps=float(1000/np.median(combo)),p95_equivalent_fps=float(1000/np.percentile(combo,95)))

def run(mode,out,repeat,include_greedy):
    out.mkdir(parents=True,exist_ok=True)
    samples=read_dataset(mode)
    attacks=get_attacks()
    detail=[];summary=[];times=[];key_checks=[];per_attack=[]
    active={name:v for name,v in VARIANTS.items() if include_greedy or name!='nonoptimal_greedy_control'}
    # Fixed train/test boundary: no tuning on reported attack data.
    for name in active:
        method=variant_method(name)
        clean=[];tested=[];per_type={};failures=0
        for h,w,host,wm in samples:
            embedded=method.embed(host,wm,key=KEYS[0])
            if embedded.side_info is not None:raise AssertionError('nonblind variant')
            stego=embedded.image
            q_psnr=float(psnr(host,stego));q_ssim=float(ssim(host,stego))
            for attack,kw in attacks:
                attacked=apply_attack(attack,stego,**kw)
                decoded=method.extract(attacked,key=KEYS[0],watermark_shape=wm.shape).watermark
                error=float(ber(wm,decoded));corr=float(nc(wm,decoded))
                row=dict(variant=name,host=h,watermark=w,attack=attack,parameters=repr(kw),PSNR=q_psnr,SSIM=q_ssim,BER=error,NC=corr,side_information_bits=0)
                detail.append(row)
                if attack=='clean':
                    clean.append(row)
                    if error!=0:failures+=1
                else:
                    tested.append(row)
                    per_type.setdefault(attack,[]).append(row)
        if failures:raise AssertionError(f'{name}: {failures} clean extraction failures')
        summary.append(dict(variant=name,host_watermark_pairs=len(samples),attacked_cases=len(tested),min_psnr_db=min(x['PSNR'] for x in clean),mean_psnr_db=float(np.mean([x['PSNR'] for x in clean])),mean_ssim=float(np.mean([x['SSIM'] for x in clean])),mean_ber=float(np.mean([x['BER'] for x in tested])),mean_nc=float(np.mean([x['NC'] for x in tested])),worst_ber=max(x['BER'] for x in tested),fifth_percentile_nc=float(np.percentile([x['NC'] for x in tested],5)),clean_ber_max=max(x['BER'] for x in clean),clean_failures=failures))
        for attack,rows in per_type.items():
            per_attack.append(dict(variant=name,attack=attack,cases=len(rows),mean_ber=float(np.mean([r['BER'] for r in rows])),mean_nc=float(np.mean([r['NC'] for r in rows]))))
        h,w,host,wm=samples[0]
        timing=speed_test(method,host,wm,KEYS[0],repeat if name!='nonoptimal_greedy_control' else min(5,repeat))
        times.append(dict(variant=name,repeats=repeat if name!='nonoptimal_greedy_control' else min(5,repeat),**timing))
        print(f'{name:28} BER={summary[-1]["mean_ber"]:.5f} PSNR_min={summary[-1]["min_psnr_db"]:.3f} FPS={timing["combined_fps"]:.1f}',flush=True)
        # Produce incremental outputs for long-running Kaggle notebooks.
        write_csv(out/'attack_rows.csv',detail,detail[0].keys())
        write_csv(out/'summary.csv',summary,summary[0].keys())
        write_csv(out/'runtime.csv',times,times[0].keys())
        write_csv(out/'by_attack.csv',per_attack,per_attack[0].keys())
    # all five keys for the final candidate; clean-image quality separately audited
    final=variant_method('FINAL_GREEN_T66')
    for k,key in enumerate(KEYS):
        for h,w,host,wm in samples:
            embedded=final.embed(host,wm,key=key)
            recovered=final.extract(embedded.image,key=key,watermark_shape=wm.shape).watermark
            row=dict(key_id=k,host=h,watermark=w,PSNR=float(psnr(host,embedded.image)),SSIM=float(ssim(host,embedded.image)),clean_BER=float(ber(wm,recovered)),side_information_bits=0)
            key_checks.append(row)
            if row['clean_BER']:raise AssertionError('multi-key clean extraction failed')
    write_csv(out/'multikey_clean.csv',key_checks,key_checks[0].keys())
    env=dict(mode=mode,host_files=sorted({h for h,w,host,wm in samples}),watermarks=sorted({w for h,w,host,wm in samples}),tested_key_count=len(KEYS),attack_conditions_per_pair=len(attacks),python=sys.version,platform=platform.platform(),numpy=np.__version__,opencv=cv2.__version__,timing_repeats=repeat,proposal_count=1,warning='Only same-hardware timing comparisons are valid; unknown-geometry robustness is not claimed.')
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    chosen=summary[0];clock=times[0]
    gates=dict(clean_ber_zero=(chosen['clean_ber_max']==0),minimum_psnr_gt50=(min(x['PSNR'] for x in key_checks)>50),combined_median_fps_gt175=(clock['combined_fps']>175),combined_p95_ms_lt_5_714=(clock['combined_p95_ms']<1000/175),mean_ber_lt_0_24=(chosen['mean_ber']<0.24))
    (out/'acceptance.json').write_text(json.dumps(dict(gates=gates,observed=chosen,latency=clock,note='These gates apply to this sample/hardware only; not evidence of held-out generalization.'),indent=2))
    return gates

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['quick','full'],default='quick');p.add_argument('--output',type=Path,default=Path('results/green_quad_only'));p.add_argument('--repeats',type=int,default=50);p.add_argument('--include-greedy-control',action='store_true');a=p.parse_args()
    if a.repeats<3: p.error('repeats >=3')
    print(json.dumps(run(a.mode,a.output,a.repeats,a.include_greedy_control),indent=2))
