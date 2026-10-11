"""Immutable host-level partition, independently auditable from its JSON manifest."""
from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
import cv2, numpy as np


def split_hosts(paths, seed=3101):
    paths=sorted(map(Path,paths),key=lambda p:p.name)
    if len(paths)<3:
        raise ValueError('Independent train/validation/test requires >=3 unique host files')
    if len({p.name for p in paths})!=len(paths):
        raise ValueError('Duplicate host basenames')
    # Stable, independent of Python hash randomization and input listing order.
    ordered=sorted(paths,key=lambda p:hashlib.sha256(f'{seed}/{p.name}'.encode()).hexdigest())
    n=len(ordered);n_train=max(1,n//3);n_val=max(1,(n-n_train)//3)
    if n_train+n_val>=n:
        raise AssertionError('No test hosts')
    return {'train':ordered[:n_train], 'validation':ordered[n_train:n_train+n_val],
            'test':ordered[n_train+n_val:]}


def manifest(split, watermarks, attacks_config, output, seed=3101):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    file_paths=[p for paths in split.values() for p in paths]+list(map(Path,watermarks))
    content={'host_split_seed':seed,'hosts':{k:[p.name for p in v] for k,v in split.items()},
             'file_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in file_paths},
             'attacks_config':str(attacks_config),'python':sys.version,'platform':platform.platform(),
             'opencv':cv2.__version__,'numpy':np.__version__,
             'protocol':'Tune parameters ONLY with train/validation. Evaluate final attacks on held-out test hosts.'}
    (output/'protocol_manifest.json').write_text(json.dumps(content,indent=2),encoding='utf8')
    return content
