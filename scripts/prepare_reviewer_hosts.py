"""Prepare REAL external host images for reproducible independent testing.

Example:
 python scripts/prepare_reviewer_hosts.py --input data/hosts/classical \
     --input /kaggle/input/extra-real-images --output /kaggle/working/hosts12

This never generates synthetic hosts. Sources must represent distinct scenes.
Perceptual near-duplicates are *flagged*, not silently counted as independent.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import cv2
import numpy as np

EXTS={'.bmp','.png','.jpg','.jpeg','.tiff','.tif'}

def normalize(img, size=512):
    """Preserve aspect ratio, scale shortest side to size, then center-crop."""
    h,w=img.shape[:2]
    if h<1 or w<1:raise ValueError('Empty image')
    factor=size/min(h,w)
    interp=cv2.INTER_AREA if factor<1 else cv2.INTER_CUBIC
    resized=cv2.resize(img,(max(size,round(w*factor)),max(size,round(h*factor))),
                       interpolation=interp)
    rh,rw=resized.shape[:2]
    y=(rh-size)//2;x=(rw-size)//2
    return resized[y:y+size,x:x+size].copy()

def difference_hash(im):
    grey=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    small=cv2.resize(grey,(9,8),interpolation=cv2.INTER_AREA)
    bits=(small[:,1:]>small[:,:-1]).reshape(-1)
    return sum(int(v)<<i for i,v in enumerate(bits))

def prepare(input_dirs,output,minimum=12):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    source_paths=sorted({p.resolve() for directory in input_dirs
                         for p in Path(directory).rglob('*')
                         if p.is_file() and p.suffix.lower() in EXTS
                         and output not in p.resolve().parents})
    records=[];hashes={};phashes=[]
    for path in source_paths:
        im=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if im is None:continue
        img=normalize(im)
        pixsha=hashlib.sha256(img.tobytes()).hexdigest()
        if pixsha in hashes:continue
        hashes[pixsha]=str(path)
        dh=difference_hash(img)
        near=[q for q,h in phashes if (h^dh).bit_count()<=5]
        phashes.append((path.name,dh))
        target=output/f'host_{len(records)+1:03d}_{pixsha[:10]}.png'
        if not cv2.imwrite(str(target),img):raise OSError(target)
        records.append({'host_file':target.name,'source_path':str(path),
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'pixel_sha256':pixsha,'input_width':int(im.shape[1]),
            'input_height':int(im.shape[0]),'near_duplicate_candidates':near,
            'normalization':'aspect_preserving_scale_short_side_512_center_crop'})
    (output/'source_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    with (output/'source_manifest.csv').open('w',newline='',encoding='utf8') as f:
        if records:
            writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    if len(records)<minimum:
        raise ValueError(f'Only {len(records)} distinct images; need >= {minimum}. '
                         'Supply genuine additional independent images (not augmentations).')
    print(f'{len(records)} unique decoded real images -> {output}; '
          'review source_manifest.csv for scene-level near-duplicates')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',action='append',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--minimum',type=int,default=12)
    args=ap.parse_args();prepare(args.input,args.output,args.minimum)
