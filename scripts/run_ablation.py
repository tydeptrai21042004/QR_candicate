from __future__ import annotations

import argparse
from pathlib import Path
from copy import deepcopy
import pandas as pd

from _common import load_yaml, resolve
from qrwatermark.core.config import load_yaml as load_core_yaml, proposed_from_dict
from qrwatermark.proposed import ProposedAdaptiveQR
from qrwatermark.evaluation.evaluator import evaluate_once
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.watermark import prepare_binary_watermark


def deep_update(base, patch):
    out=deepcopy(base)
    for k,v in patch.items():
        if isinstance(v,dict) and isinstance(out.get(k),dict): out[k]=deep_update(out[k],v)
        else: out[k]=v
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/experiments/ablation.yaml'); ap.add_argument('--key',required=True); ap.add_argument('--run-dir',required=True); args=ap.parse_args()
    cfg=load_yaml(args.config); base=load_core_yaml(resolve(cfg['base_config']))['parameters']; rows=[]
    hp=resolve('data/hosts/classical/girl.bmp'); host=read_color(hp)
    for wp in sorted(resolve('data/watermarks').glob('*.png')):
        wm=prepare_binary_watermark(wp,64)
        for name,patch in cfg['variants'].items():
            method=ProposedAdaptiveQR(proposed_from_dict(deep_update(base,patch)))
            for attack,params in [('clean',{}),('gaussian_noise',{'variance':0.003,'mean':0.0}),('jpeg',{'quality':50}),('salt_pepper',{'density':0.1})]:
                rec,_=evaluate_once(method,host,wm,key=args.key.encode(),host_name=hp.stem,watermark_name=wp.stem,attack_name=attack,attack_params=params,seed=0 if attack in {'gaussian_noise','salt_pepper'} else None)
                row=rec.__dict__; row['variant']=name; rows.append(row)
    rd=Path(args.run_dir); rd.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(rd/'ablation.csv',index=False)
    print(f'Completed {len(rows)} ablation cases')
if __name__=='__main__': main()
