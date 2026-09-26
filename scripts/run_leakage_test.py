from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

from _common import load_yaml, resolve
from qrwatermark.core.factory import build_method
from qrwatermark.proposed.side_info import unpack_modes
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.watermark import prepare_binary_watermark, arnold_transform, bits_from_watermark


def mutual_information_binary(x: np.ndarray, y: np.ndarray) -> float:
    x=np.asarray(x,dtype=np.uint8).ravel(); y=np.asarray(y,dtype=np.uint8).ravel()
    n=len(x); mi=0.0
    for a in (0,1):
        for b in (0,1):
            pxy=np.mean((x==a)&(y==b))
            if pxy<=0: continue
            px=np.mean(x==a); py=np.mean(y==b)
            mi += pxy*math.log2(pxy/(px*py))
    return float(mi)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/experiments/leakage.yaml')
    ap.add_argument('--key',required=True)
    ap.add_argument('--run-dir',required=True)
    args=ap.parse_args()
    cfg=load_yaml(args.config); m=cfg['method']
    method=build_method(m['name'],resolve(m['config']))
    rows=[]; all_modes=[]; all_bits=[]
    key=args.key.encode()
    for wp in sorted(resolve(cfg['watermarks']).glob('*.png')):
        wm=prepare_binary_watermark(wp,64)
        scrambled=arnold_transform(wm,method.config.arnold_iterations)
        bits=bits_from_watermark(scrambled)
        for hp in sorted(resolve(cfg['hosts']).glob('*.bmp')):
            host=read_color(hp)
            emb=method.embed(host,wm,key=key)
            modes,_=unpack_modes(emb.side_info,key)
            all_modes.append(modes); all_bits.append(bits)
            p_q0=float(np.mean(modes[bits==0]==0)) if np.any(bits==0) else float('nan')
            p_q1=float(np.mean(modes[bits==1]==0)) if np.any(bits==1) else float('nan')
            rows.append({'host':hp.stem,'watermark':wp.stem,'p_q_given_0':p_q0,'p_q_given_1':p_q1,'delta':abs(p_q1-p_q0),'mi_bits':mutual_information_binary(modes,bits)})
    modes=np.concatenate(all_modes); bits=np.concatenate(all_bits)
    # Best one-bit mapping fitted only as a leakage diagnostic, never used by extraction.
    pred_a=(modes==0).astype(np.uint8)
    pred_b=1-pred_a
    ber_flag=min(float(np.mean(pred_a!=bits)),float(np.mean(pred_b!=bits)))
    summary={'aggregate_p_q_given_0':float(np.mean(modes[bits==0]==0)),'aggregate_p_q_given_1':float(np.mean(modes[bits==1]==0)),'aggregate_delta':abs(float(np.mean(modes[bits==1]==0))-float(np.mean(modes[bits==0]==0))),'aggregate_mi_bits':mutual_information_binary(modes,bits),'best_flag_only_ber':ber_flag,'samples':int(len(bits))}
    rd=Path(args.run_dir); rd.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(rd/'leakage_per_case.csv',index=False)
    pd.DataFrame([summary]).to_csv(rd/'leakage_summary.csv',index=False)
    print(summary)
if __name__=='__main__': main()
