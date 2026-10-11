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
             'protocol':'Tune parameters ONLY with train/validation. Evaluate final attacks on held-out test hosts.',
             'n_independent_hosts':len(file_paths)-len(watermarks),
             'n_train':len(split['train']), 'n_validation':len(split['validation']),
             'n_test':len(split['test'])}
    (output/'protocol_manifest.json').write_text(json.dumps(content,indent=2),encoding='utf8')
    return content


def audit_host_corpus(paths, *, min_hosts=12, require_shape=(512,512)):
    """Reject duplicate images and noncompliant inputs before splitting hosts.

    Byte-level and decoded-pixel identity both count as duplicates; filenames
    alone are not evidence of independent host images. Source provenance and
    near-duplicate semantic checking remain the experimenter's responsibility.
    """
    paths = [Path(p) for p in paths]
    if len(paths) < min_hosts:
        raise ValueError(f'Need >= {min_hosts} independent REAL host images; found {len(paths)}. '
                         'Supply additional images with --hosts-dir / --extra-hosts-dir '
                         '(or explicitly use --min-hosts 6 for exploratory results).')
    if len({p.stem.lower() for p in paths}) != len(paths):
        raise ValueError('Duplicate host stems would collide in evaluation identifiers')
    seen_bytes={}; seen_pixels={}; checks=[]
    for path in paths:
        blob=path.read_bytes()
        raw_sha=hashlib.sha256(blob).hexdigest()
        if raw_sha in seen_bytes:
            raise ValueError(f'Duplicate host file: {path} and {seen_bytes[raw_sha]}')
        seen_bytes[raw_sha]=str(path)
        im=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if im is None or im.ndim != 3 or im.shape[2]!=3:
            raise ValueError(f'Unreadable RGB/color image: {path}')
        if require_shape and tuple(im.shape[:2])!=tuple(require_shape):
            raise ValueError(f'{path}: expected {require_shape} pixels, got {im.shape[:2]}; '
                             'prepare a consistent host corpus before running')
        pixel_sha=hashlib.sha256(im.tobytes()).hexdigest()
        if pixel_sha in seen_pixels:
            raise ValueError(f'Duplicate decoded host pixels: {path} and {seen_pixels[pixel_sha]}')
        seen_pixels[pixel_sha]=str(path)
        checks.append({'name':path.name,'path':str(path),'width':int(im.shape[1]),
                       'height':int(im.shape[0]),'file_sha256':raw_sha,
                       'pixel_sha256':pixel_sha})
    return checks
