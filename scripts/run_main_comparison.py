from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import replace

from _common import load_yaml, resolve
from qrwatermark.core.factory import build_method
from qrwatermark.evaluation.aggregation import aggregate_records
from qrwatermark.evaluation.review_protocol import audit_host_corpus
from qrwatermark.evaluation.review_statistics import statistical_report
from qrwatermark.evaluation.benchmark import run_benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/main_comparison.yaml")
    ap.add_argument("--key", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--hosts-dir",type=Path,default=None,
                    help="Override host corpus with independent prepared 512x512 hosts")
    ap.add_argument("--min-hosts",type=int,default=12,
                    help="12 required for principal evidence; 6 explicitly exploratory")
    ap.add_argument("--psnr-tolerance",type=float,default=0.2)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    watermark_size = int(cfg.get("watermark_size", 64))
    methods = [build_method(m["name"], resolve(m["config"])) for m in cfg["methods"]]
    # Experiment-level payload size overrides the method YAML so a small-payload
    # control study (e.g. Zareian2013: one bit per 16x16 host block) remains an
    # apples-to-apples comparison across every method.
    for method in methods:
        if hasattr(method, "config") and hasattr(method.config, "watermark_size"):
            method.config = replace(method.config, watermark_size=watermark_size)
    host_dir=args.hosts_dir or resolve(cfg["hosts"])
    hosts=sorted(p for p in host_dir.iterdir() if p.suffix.lower() in
                 ('.bmp','.png','.jpg','.jpeg','.tiff','.tif') and p.is_file())
    audit=audit_host_corpus(hosts,min_hosts=args.min_hosts)
    run_dir=Path(args.run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    (run_dir/'host_audit.json').write_text(json.dumps({
        'hosts':audit,'n_hosts':len(hosts),'exploratory_under_12':len(hosts)<12,
        'design':'Fixed operating-point comparison without PSNR matching. '
                 'PSNR is measured and reported, not used as an eligibility threshold. '
                 'This corpus is descriptive; do not call it independent held-out validation.'},indent=2))
    watermarks = sorted(resolve(cfg["watermarks"]).glob("*.png"))
    attacks = load_yaml(cfg["attacks"])["attacks"]
    df = run_benchmark(
        methods, hosts, watermarks, attacks,
        key=args.key.encode(),
        run_dir=Path(args.run_dir),
        save_images=args.save_images,
        match_psnr_to_proposed=bool(cfg.get("match_psnr_to_proposed", True)),
        watermark_size=watermark_size,
    )
    aggregate_records(df).to_csv(Path(args.run_dir) / "summary.csv", index=False)
    # A PSNR-target flag is NOT evidence that matching succeeded.
    ref='green_quad_min_energy'
    clean=df.loc[df.attack.eq('clean'),['method','host','watermark','embedding_psnr']].copy()
    proposed=clean.loc[clean.method.eq(ref),['host','watermark','embedding_psnr']].rename(
        columns={'embedding_psnr':'reference_psnr_db'})
    audit_df=clean.merge(proposed,on=['host','watermark'],validate='many_to_one')
    audit_df['psnr_gap_db']=audit_df.embedding_psnr-audit_df.reference_psnr_db
    audit_df['within_tolerance']=audit_df.psnr_gap_db.abs().le(args.psnr_tolerance)
    audit_df.to_csv(run_dir/'psnr_audit.csv',index=False)
    audit_df.groupby('method',as_index=False).agg(
        n_pairs=('within_tolerance','size'),n_matched=('within_tolerance','sum'),
        mean_abs_gap_db=('psnr_gap_db',lambda x:float(np.abs(x).mean()))
        ).to_csv(run_dir/'psnr_coverage.csv',index=False)
    eligible=audit_df[['method','host','watermark','within_tolerance']]
    eligible_df=df.merge(eligible,on=['method','host','watermark'],validate='many_to_one')
    eligible_df.to_csv(run_dir/'metrics_with_eligibility.csv',index=False)
    statistical_report(eligible_df,run_dir/'unrestricted',pair_policy='strict')
    # Optional secondary diagnostic; the operating-point comparison never
    # requires achieving a fixed PSNR tolerance.
    matched_diagnostic=bool(cfg.get('produce_matched_diagnostic',False))
    if matched_diagnostic:
        matched=eligible_df.loc[eligible_df.within_tolerance]
        if len(matched):
            statistical_report(matched,run_dir/'matched_only',pair_policy='intersection',
                               expected_methods=[m.name for m in methods])
    (run_dir/'matching_protocol.json').write_text(json.dumps({
        'psnr_tolerance_db':args.psnr_tolerance,
        'study':'unmatched_operating_point_comparison',
        'psnr_matching_required':False,
        'produce_matched_diagnostic':matched_diagnostic,
        'matched_comparisons':'intersection of exact shared evaluation identifiers',
        'unmatched':'excluded from matched statistical comparisons',
        'caution':'PSNR is reported alongside BER and NC for transparency. '
                  'Unmatched operating-point results do not demonstrate superiority at equal distortion.'},indent=2))
    print(f"Completed {len(df)} evaluations -> {args.run_dir}")


if __name__ == "__main__":
    main()
