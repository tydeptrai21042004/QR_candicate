from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from _common import load_yaml, resolve
from qrwatermark.core.config import load_yaml as load_core_yaml, proposed_from_dict
from qrwatermark.proposed import ProposedAdaptiveQR
from qrwatermark.evaluation.evaluator import evaluate_once
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.watermark import prepare_binary_watermark


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/experiments/repetition.yaml'); ap.add_argument('--key',required=True); ap.add_argument('--run-dir',required=True); args=ap.parse_args()
    cfg=load_yaml(args.config); base=load_core_yaml(resolve(cfg['base_config']))['parameters']
    rows=[]
    for r in cfg['repetitions']:
        d=dict(base); d['repetition']=int(r); method=ProposedAdaptiveQR(proposed_from_dict(d))
        for wp in sorted(resolve('data/watermarks').glob('*.png')):
            wm=prepare_binary_watermark(wp,64)
            for hp in sorted(resolve('data/hosts/classical').glob('*.bmp')):
                host=read_color(hp)
                rec,_=evaluate_once(method,host,wm,key=args.key.encode(),host_name=hp.stem,watermark_name=wp.stem,attack_name='clean')
                row=rec.__dict__; row['repetition']=r; rows.append(row)
    rd=Path(args.run_dir); rd.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(rd/'repetition_study.csv',index=False)
    print(f'Completed {len(rows)} cases')
if __name__=='__main__': main()
