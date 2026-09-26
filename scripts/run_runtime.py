from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from _common import load_yaml, resolve
from qrwatermark.core.factory import build_method
from qrwatermark.evaluation.evaluator import evaluate_once
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.watermark import prepare_binary_watermark


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/experiments/runtime.yaml'); ap.add_argument('--key',required=True); ap.add_argument('--run-dir',required=True); args=ap.parse_args()
    cfg=load_yaml(args.config); methods=[build_method(m['name'],resolve(m['config'])) for m in cfg['methods']]
    host=read_color(resolve('data/hosts/classical/girl.bmp')); wm=prepare_binary_watermark(resolve('data/watermarks/watermark_1.png'),64); rows=[]
    for method in methods:
        for rep in range(int(cfg.get('repeats',3))):
            rec,_=evaluate_once(method,host,wm,key=args.key.encode(),host_name='girl',watermark_name='watermark_1',attack_name='clean')
            row=rec.__dict__; row['runtime_repeat']=rep; rows.append(row)
    rd=Path(args.run_dir); rd.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(rd/'runtime.csv',index=False)
    print(f'Completed {len(rows)} runtime measurements')
if __name__=='__main__': main()
