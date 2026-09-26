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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/attack_sweep.yaml")
    ap.add_argument("--key", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config); m = cfg["method"]
    method = build_method(m["name"], resolve(m["config"]))
    hosts = sorted((resolve("data/hosts/classical")).glob("*.bmp"))
    wm_paths = sorted((resolve("data/watermarks")).glob("*.png"))
    rows=[]
    for wp in wm_paths:
        wm=prepare_binary_watermark(wp,64)
        for hp in hosts:
            host=read_color(hp)
            for attack, spec in cfg["attack_sweeps"].items():
                param=spec["parameter"]
                for value in spec["values"]:
                    for seed in spec.get("seeds",[None]):
                        rec,_=evaluate_once(method,host,wm,key=args.key.encode(),host_name=hp.stem,watermark_name=wp.stem,attack_name=attack,attack_params={param:value},seed=seed)
                        row=rec.__dict__; row["sweep_parameter"]=param; row["sweep_value"]=value; rows.append(row)
    rd=Path(args.run_dir); rd.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(rd/"attack_sweep.csv",index=False)
    print(f"Completed {len(rows)} sweep evaluations")
if __name__ == "__main__": main()
