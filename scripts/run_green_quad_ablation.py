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
from qrwatermark.evaluation.review_protocol import audit_host_corpus, split_hosts

HOSTS=('airplane','girl','lenna','manhattan','pepper','safari')
WMS=('watermark_1','watermark_2')
KEYS=(b'reviewer-speed-recheck-2026',b'independent-key-B',b'independent-key-C',b'independent-key-D',b'independent-key-E')
# Scientific separation: structural/component controls vs parameter selection.
# Neither study modifies the one registered production algorithm.
FINAL = {'FINAL_GREEN_T66': {}}
ABLATION_VARIANTS = {
    **FINAL,
    'channel_blue': {'channel': 0},
    'channel_red': {'channel': 2},
    'nonoptimal_greedy_control': {'projection': 'greedy_control'},
}
HYPERPARAMETER_VARIANTS = {
    **FINAL,
    **{f'period_{t}': {'integer_conv4_period': t} for t in (60, 62, 64, 68, 70, 72)},
    'arnold_05': {'arnold_iterations': 5},
    'arnold_10': {'arnold_iterations': 10},
}
STUDIES = {'ablation': ABLATION_VARIANTS, 'hyperparameters': HYPERPARAMETER_VARIANTS}
# Backward-compatibility for prior imports; the default CLI runs ablations only.
VARIANTS = ABLATION_VARIANTS

def write_csv(path,rows,header):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=header);writer.writeheader();writer.writerows(rows)

def variant_method(name, study='ablation'):
    cfg=ProposedConfig(**{'ablation_mode':name!='FINAL_GREEN_T66',**STUDIES[study][name]})
    return GreenQuadWatermark(cfg)

def read_dataset(mode, hosts_dir=None, partition='test', min_hosts=6, seed=3101):
    """Deterministically partition independent hosts *before* measuring anything.

    Use validation hosts for hyperparameter selection and held-out test hosts
    for architectural ablations. The in-repository six-host corpus is explicitly
    exploratory; a supplied 12-host corpus supports the main reviewer protocol.
    """
    host_dir = Path(hosts_dir) if hosts_dir else ROOT/'data'/'hosts'/'classical'
    paths = sorted(p for p in host_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in ('.bmp','.png','.jpg','.jpeg','.tif','.tiff'))
    checks = audit_host_corpus(paths, min_hosts=min_hosts)
    groups = split_hosts(paths, seed)
    if partition not in ('all', 'train', 'validation', 'test'):
        raise ValueError(f'Unknown dataset partition: {partition}')
    selected = paths if partition == 'all' else groups[partition]
    if mode == 'quick':
        selected = selected[:1]
    if not selected: raise ValueError(f'No hosts in partition {partition}')
    wms = WMS if mode == 'full' else WMS[:1]
    data = []
    for path in selected:
        host = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if host is None: raise FileNotFoundError(path)
        for w in wms:
            wm=prepare_binary_watermark(ROOT/'data'/'watermarks'/f'{w}.png')
            data.append((path.stem,w,host,wm))
    protocol = {'seed': seed, 'partition': partition,
                'n_independent_hosts': len(checks),
                'exploratory_under_12': len(checks)<12,
                'host_groups': {k:[p.name for p in v] for k,v in groups.items()},
                'evaluated_hosts': [p.name for p in selected],
                'n_watermarks':len(wms), 'mode':mode,
                'warning': 'Hyperparameters must be selected on validation ONLY; do not tune on test hosts.'}
    return data, protocol

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

def run(mode,out,repeat,include_greedy,*,study="ablation",hosts_dir=None,partition=None,min_hosts=6):
    out.mkdir(parents=True,exist_ok=True)
    if study not in STUDIES: raise ValueError(f'Unknown study: {study}')
    if partition is None: partition = 'validation' if study=='hyperparameters' else 'test'
    samples,protocol=read_dataset(mode,hosts_dir=hosts_dir,partition=partition,min_hosts=min_hosts)
    (out/'study_protocol.json').write_text(json.dumps({**protocol,'study':study,
        'variants':STUDIES[study],'fixed_baseline':'FINAL_GREEN_T66',
        'same_attack_grid_for_all_variants':True},indent=2))
    attacks=get_attacks()
    detail=[];summary=[];times=[];key_checks=[];per_attack=[]
    active={name:v for name,v in STUDIES[study].items()
            if include_greedy or name!='nonoptimal_greedy_control'}
    # Fixed train/test boundary: no tuning on reported attack data.
    for name in active:
        method=variant_method(name,study=study)
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
    # Explicitly separate validation-only hyperparameter selection from final
    # held-out ablation reporting. Never rewrite the production T=66 config.
    if study == 'hyperparameters':
        eligible = [r for r in summary if r['min_psnr_db'] > 50
                    and r['clean_ber_max'] == 0]
        # The machine-specific speed criterion is included only as a report;
        # it must not turn an ineligible test into a paper claim.
        rank = sorted(eligible, key=lambda r:(r['mean_ber'],-r['mean_nc'],
                                               -r['min_psnr_db'],r['variant']))
        selected = rank[0]['variant'] if rank else None
        (out/'hyperparameter_selection.json').write_text(json.dumps({
            'partition':partition, 'selected_on_validation_only':selected,
            'production_parameter_unchanged':'FINAL_GREEN_T66',
            'selection_rule':'minimize mean BER subject to PSNR_min > 50 dB and clean BER=0',
            'fps_reporting':'Measured separately and must be checked on the final hardware',
            'eligible_variants':[r['variant'] for r in eligible],
            'warning':'Selection is exploratory with <12 hosts and is NOT a test-set result.'
                if protocol['exploratory_under_12'] else
                'Freeze the selected parameters before any test-host evaluation.'},indent=2))
    else:
        final_row = next(r for r in summary if r['variant']=='FINAL_GREEN_T66')
        delta=[]
        for r in summary:
            delta.append(dict(variant=r['variant'],
                delta_mean_ber=r['mean_ber']-final_row['mean_ber'],
                delta_mean_nc=r['mean_nc']-final_row['mean_nc'],
                delta_min_psnr_db=r['min_psnr_db']-final_row['min_psnr_db']))
        write_csv(out/'ablation_deltas.csv',delta,delta[0].keys())
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
    env=dict(study=study,partition=partition,independent_host_count=protocol['n_independent_hosts'],mode=mode,host_files=sorted({h for h,w,host,wm in samples}),watermarks=sorted({w for h,w,host,wm in samples}),tested_key_count=len(KEYS),attack_conditions_per_pair=len(attacks),python=sys.version,platform=platform.platform(),numpy=np.__version__,opencv=cv2.__version__,timing_repeats=repeat,proposal_count=1,warning='Only same-hardware timing comparisons are valid; unknown-geometry robustness is not claimed.')
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    chosen=summary[0];clock=times[0]
    gates=dict(clean_ber_zero=(chosen['clean_ber_max']==0),minimum_psnr_gt50=(min(x['PSNR'] for x in key_checks)>50),combined_median_fps_gt175=(clock['combined_fps']>175),combined_p95_ms_lt_5_714=(clock['combined_p95_ms']<1000/175),mean_ber_lt_0_24=(chosen['mean_ber']<0.24))
    (out/'acceptance.json').write_text(json.dumps(dict(gates=gates,observed=chosen,latency=clock,note='These gates apply to this sample/hardware only; not evidence of held-out generalization.'),indent=2))
    return gates

def main(default_study='ablation'):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['quick','full'],default='quick')
    p.add_argument('--study',choices=tuple(STUDIES),default=default_study)
    p.add_argument('--output',type=Path,default=Path('results/green_quad_only'))
    p.add_argument('--repeats',type=int,default=50)
    p.add_argument('--include-greedy-control',action='store_true')
    p.add_argument('--hosts-dir',type=Path,default=None)
    p.add_argument('--min-hosts',type=int,default=6)
    p.add_argument('--partition',choices=('train','validation','test','all'),default=None)
    a=p.parse_args()
    if a.repeats<3:p.error('repeats >=3')
    print(json.dumps(run(a.mode,a.output,a.repeats,a.include_greedy_control,
        study=a.study,hosts_dir=a.hosts_dir,partition=a.partition,
        min_hosts=a.min_hosts),indent=2))

if __name__=='__main__':
    main()

