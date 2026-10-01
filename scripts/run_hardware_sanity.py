from __future__ import annotations

import argparse
import numpy as np

from qrwatermark.proposed.qr import canonical_qr
from qrwatermark.proposed.r_branch import apply_r12_delta, qim_phase_decision


def main():
    ap=argparse.ArgumentParser(description="Lightweight equivalence checks for the v4 hardware datapath")
    ap.add_argument("--samples",type=int,default=20000)
    ap.add_argument("--seed",type=int,default=2026)
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed)

    max_error=0.0; rounded_mismatches=0
    for _ in range(args.samples):
        block=rng.uniform(8.0,247.0,size=(2,2))
        delta=float(rng.uniform(-10.0,10.0))
        q,r=canonical_qr(block); rr=r.copy(); rr[0,1]+=delta
        reference=q@rr; fast=apply_r12_delta(block,delta)
        max_error=max(max_error,float(np.max(np.abs(reference-fast))))
        rounded_mismatches+=int(not np.array_equal(np.rint(reference),np.rint(fast)))

    period=36.0; qim_mismatches=0
    for stat in rng.uniform(-2000.0,2000.0,size=args.samples):
        phase=float(stat)%period
        if min(phase,abs(phase-period/2.0),period-phase)<1e-10:
            continue
        old=int(np.sin(2.0*np.pi*(phase-0.5*period)/period)>=0.0)
        new,_,_=qim_phase_decision(float(stat),period)
        qim_mismatches+=int(old!=new)

    print(f"samples={args.samples}")
    print(f"max_closed_form_error={max_error:.3e}")
    print(f"rounded_reconstruction_mismatches={rounded_mismatches}")
    print(f"qim_hard_decision_mismatches={qim_mismatches}")
    if rounded_mismatches or qim_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
