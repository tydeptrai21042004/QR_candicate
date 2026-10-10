"""Run FCQR-v6 versus experimental 8-pixel integer convolution on real images.

Usage: PYTHONPATH=src python scripts/benchmark_integer_conv_v7.py --out evidence_integer_v7
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

from benchmark_fcqr_v6 import ATTACKS
from qrwatermark.attacks.registry import apply_attack
from qrwatermark.core.config import ProposedConfig
from qrwatermark.core.factory import build_method
from qrwatermark.evaluation.metrics import ber, nc, psnr
from qrwatermark.utils.watermark import prepare_binary_watermark


def speed(method, host, watermark, key, n=60):
    embedded=method.embed(host,watermark,key=key).image
    fn={'embed': lambda: method.embed(host,watermark,key=key),
        'extract':lambda: method.extract(embedded,key=key)}
    output={}
    for name,call in fn.items():
        for _ in range(8):call()
        data=[]
        for _ in range(n):
            t=time.perf_counter();call();data.append((time.perf_counter()-t)*1000)
        tracemalloc.start();call();_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        output[name]={'median_ms':statistics.median(data),
                      'peak_python_alloc_mib':peak/1048576}
    output['fps']=1000/(output['embed']['median_ms']+output['extract']['median_ms'])
    return output


def run(root,out,n=60):
    cv2.setNumThreads(1)
    hosts=sorted((root/'data/hosts/classical').glob('*.bmp'))
    wms=sorted((root/'data/watermarks').glob('*.png'))
    if len(hosts)!=6 or len(wms)!=2: raise RuntimeError('expected six host images and two watermark images')
    assert len(ATTACKS)==26
    out.mkdir(parents=True,exist_ok=True)
    methods={
      'FCQR_v6': build_method('proposed','configs/methods/proposed_fcqr_v6_experimental.yaml'),
      'IntegerConv8_v7': build_method('proposed','configs/methods/proposed_integer_conv8_v7_experimental.yaml'),
      'IntegerConvPair_v8': build_method('proposed','configs/methods/proposed_integer_conv_pair_v8_experimental.yaml'),
      'IntegerConvQuad_v9': build_method('proposed','configs/methods/proposed_integer_conv_quad_v9_experimental.yaml'),
    }
    key=b'fcqr_v6_same_fixed_key_for_both_methods'
    rows=[];summary={}
    for name,method in methods.items():
        clean_psnr=[];clean_ber=[];attacked_nc=[];attacked_ber=[]
        for hp in hosts:
            host=cv2.imread(str(hp))
            for wp in wms:
                watermark=prepare_binary_watermark(wp,64)
                embedded=method.embed(host,watermark,key=key)
                stego=embedded.image
                p=psnr(host,stego)
                clean_psnr.append(float(p))
                for attack,params in ATTACKS:
                    result=method.extract(apply_attack(attack,stego,**params),key=key).watermark
                    a_ber=float(ber(watermark,result));a_nc=float(nc(watermark,result))
                    rows.append({'method':name,'host':hp.stem,'watermark':wp.stem,'attack':attack,
                         'parameters':json.dumps(params,sort_keys=True),
                         'clean_psnr_db':p,'nc':a_nc,'ber':a_ber})
                    if attack=='clean':clean_ber.append(a_ber)
                    else:attacked_nc.append(a_nc);attacked_ber.append(a_ber)
        summary[name]={
            'host_count':len(hosts),'watermarks':len(wms),'attacks_per_pair':25,
            'mean_psnr_db':float(np.mean(clean_psnr)),
            'min_psnr_db':float(min(clean_psnr)),
            'max_clean_ber':float(max(clean_ber)),
            'mean_attacked_nc':float(np.mean(attacked_nc)),
            'mean_attacked_ber':float(np.mean(attacked_ber)),
        }
    host=cv2.imread(str(hosts[0]));wm=prepare_binary_watermark(wms[0],64)
    # Interleaved measurement to reduce systematic scheduling bias.
    speeds={name:[] for name in methods}
    for trial in range(5):
        for name,method in methods.items():
            speeds[name].append(speed(method,host,wm,key,n=n))
    for name in methods:
        summary[name]['runtime']={}
        for key_name in ('embed','extract'):
            summary[name]['runtime'][key_name]={k:float(statistics.median(item[key_name][k] for item in speeds[name]))
                for k in ('median_ms','peak_python_alloc_mib')}
        e=summary[name]['runtime']['embed']['median_ms'];x=summary[name]['runtime']['extract']['median_ms']
        summary[name]['runtime']['fps']=1000/(e+x)
    with (out/'per_image_25_attacks.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('evidence_integer_v7'))
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--repeats',type=int,default=60)
    args=parser.parse_args()
    run(args.root,args.out,args.repeats)
