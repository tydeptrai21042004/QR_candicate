"""Backwards-compatible entry point for the one final method plus internal ablation."""
from run_green_quad_ablation import run
import argparse
from pathlib import Path
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['quick','full'],default='quick')
    ap.add_argument('--output',type=Path,default=Path('results/green_quad_only'))
    ap.add_argument('--repeats',type=int,default=50)
    ap.add_argument('--include-greedy-control',action='store_true')
    a=ap.parse_args()
    run(a.mode,a.output,a.repeats,a.include_greedy_control)
