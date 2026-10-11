"""Apply the deletion manifest after overlaying this change-only ZIP.

Run from the repository root: python APPLY_SINGLE_PROPOSAL.py
Use --dry-run to print all targets without deleting.
This intentionally retains published baseline methods.
"""
from pathlib import Path
import argparse

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');args=p.parse_args()
expected=root/'src/qrwatermark/proposed/green_quad_min_energy.py'
if not expected.is_file():raise SystemExit('Missing updated final method; extract patch into repository root first')
manifest=root/'DELETE_FILES.txt'
if not manifest.is_file():raise SystemExit('Missing DELETE_FILES.txt')
removed=0
for line in manifest.read_text(encoding='utf8').splitlines():
    rel=line.strip()
    if not rel or rel.startswith('#'):continue
    part=Path(rel)
    if part.is_absolute() or '..' in part.parts:raise SystemExit(f'Invalid deletion target: {rel}')
    target=root/part
    if not target.is_file():continue
    if args.dry_run:print('Would delete',rel)
    else:
        target.unlink()
        removed+=1
if not args.dry_run:
    for d in ('src/qrwatermark/proposed','configs/experiments','configs/methods','tests','scripts','docs','evidence','validation'):
        directory=root/d
        if directory.is_dir():
            for sub in sorted(directory.rglob('*'),key=lambda x:len(x.parts),reverse=True):
                if sub.is_dir() and not any(sub.iterdir()):sub.rmdir()
    active={p.name for p in (root/'src/qrwatermark/proposed').glob('*.py')}
    expect={'__init__.py','block_carrier.py','method.py','green_quad_min_energy.py'}
    if active!=expect:
        raise SystemExit(f'Unexpected proposal files remain: {active^expect}')
    print(f'Cleanup complete: {removed} obsolete files deleted. Published baselines remain intact.')
