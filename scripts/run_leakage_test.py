from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

from _common import load_yaml, resolve
from qrwatermark.core.factory import build_method
from qrwatermark.proposed.side_info import unpack_period_indices,unpack_certified_mask
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.watermark import prepare_binary_watermark, arnold_transform, bits_from_watermark


def mutual_information_discrete_binary(x: np.ndarray, y: np.ndarray) -> float:
    x=np.asarray(x).ravel(); y=np.asarray(y,dtype=np.uint8).ravel()
    mi=0.0
    for a in np.unique(x):
        for b in (0,1):
            pxy=np.mean((x==a)&(y==b))
            if pxy<=0: continue
            px=np.mean(x==a); py=np.mean(y==b)
            mi += pxy*math.log2(pxy/(px*py))
    return float(mi)


def best_code_only_ber(codes: np.ndarray, bits: np.ndarray) -> float:
    pred=np.zeros_like(bits)
    for code in np.unique(codes):
        mask=codes==code
        ones=float(np.mean(bits[mask])) if np.any(mask) else 0.5
        pred[mask]=1 if ones>=0.5 else 0
    return float(np.mean(pred!=bits))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/experiments/leakage.yaml')
    ap.add_argument('--key',required=True)
    ap.add_argument('--run-dir',required=True)
    args=ap.parse_args()
    cfg=load_yaml(args.config); m=cfg['method']
    method=build_method(m['name'],resolve(m['config']))
    rows=[]; all_codes=[]; all_joint=[]; all_bits=[]
    key=args.key.encode()
    for wp in sorted(resolve(cfg['watermarks']).glob('*.png')):
        wm=prepare_binary_watermark(wp,64)
        scrambled=arnold_transform(wm,method.config.arnold_iterations)
        bits=bits_from_watermark(scrambled)
        for hp in sorted(resolve(cfg['hosts']).glob('*.bmp')):
            host=read_color(hp)
            emb=method.embed(host,wm,key=key)
            codes,_=unpack_period_indices(emb.side_info,key)
            cert=unpack_certified_mask(emb.side_info,key).astype(np.uint8)
            joint=codes.astype(np.uint16)*2+cert.astype(np.uint16)
            all_codes.append(codes); all_joint.append(joint); all_bits.append(bits)
            rows.append({
                'host':hp.stem,
                'watermark':wp.stem,
                'period_code_mi_bits':mutual_information_discrete_binary(codes,bits),
                'best_period_code_only_ber':best_code_only_ber(codes,bits),
                'joint_period_cert_mi_bits':mutual_information_discrete_binary(joint,bits),
                'best_joint_side_code_ber':best_code_only_ber(joint,bits),
                'certified_fraction':emb.metadata.get('certified_fraction'),
            })
    codes=np.concatenate(all_codes); joint=np.concatenate(all_joint); bits=np.concatenate(all_bits)
    summary={
        'aggregate_period_code_mi_bits':mutual_information_discrete_binary(codes,bits),
        'best_period_code_only_ber':best_code_only_ber(codes,bits),
        'aggregate_joint_period_cert_mi_bits':mutual_information_discrete_binary(joint,bits),
        'best_joint_side_code_ber':best_code_only_ber(joint,bits),
        'samples':int(len(bits)),
    }
    rd=Path(args.run_dir); rd.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(rd/'leakage_per_case.csv',index=False)
    pd.DataFrame([summary]).to_csv(rd/'leakage_summary.csv',index=False)
    print(summary)
if __name__=='__main__': main()
