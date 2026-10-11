"""Strict adapter for an independently installed, pretrained neural baseline.

NO neural weights are shipped in this patch. A pretrained model's official
embed/extract inference must be installed and supplied using --runner.
This adapter validates actual image and payload I/O and writes an explicit
status rather than fabricating a missing SOTA comparison.

Runner contract (any model architecture):
  python runner.py embed --input HOST --watermark WATERMARK --output STEGO
             --checkpoint FILE --key KEY
  python runner.py extract --input ATTACKED --output WATERMARK
             --checkpoint FILE --key KEY --watermark-size 64
Both outputs must be uint8 PNGs. A model that cannot carry 4096 bits at
512x512 must be reported in a SEPARATE capacity stratum, never mixed into
the fair, 4096-bit matched-PSNR study.
"""
from __future__ import annotations
import argparse,json,hashlib,subprocess,sys,tempfile,time,platform
from pathlib import Path
import numpy as np
from _common import resolve
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.watermark import prepare_binary_watermark
from qrwatermark.evaluation.metrics import ber,nc,psnr,ssim
from qrwatermark.attacks import apply_attack


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(args.checkpoint)
    runner=Path(args.runner)
    if not runner.is_file() or not checkpoint.is_file():
        raise FileNotFoundError('Actual pretrained runner script AND checkpoint are mandatory')
    if args.native_payload_bits != args.payload_size**2:
        raise ValueError('Model native capacity differs from tested payload; use a separate capacity comparison')
    host=read_color(resolve(args.host));wm=prepare_binary_watermark(resolve(args.watermark),args.payload_size)
    with tempfile.TemporaryDirectory() as tmp:
        temp=Path(tmp);src=temp/'host.png';mark=temp/'wm.png';stego=temp/'stego.png';decoded=temp/'decoded.png'
        import cv2
        cv2.imwrite(str(src),host);cv2.imwrite(str(mark),wm)
        commands=[
            [sys.executable,str(runner),'embed','--input',str(src),'--watermark',str(mark),
             '--output',str(stego),'--checkpoint',str(checkpoint),'--key',args.key],
            [sys.executable,str(runner),'extract','--input',str(stego),'--output',str(decoded),
             '--checkpoint',str(checkpoint),'--key',args.key,'--watermark-size',str(args.payload_size)]]
        timings=[]
        for command in commands:
            t0=time.perf_counter()
            subprocess.run(command,check=True,timeout=args.timeout,capture_output=True,text=True)
            timings.append(time.perf_counter()-t0)
        y=cv2.imread(str(stego),cv2.IMREAD_COLOR)
        r=cv2.imread(str(decoded),cv2.IMREAD_UNCHANGED)
        if y is None or r is None or y.shape!=host.shape or y.dtype!=np.uint8:
            raise RuntimeError('Invalid neural embed image / shape / dtype')
        if r is None or r.shape[:2]!=wm.shape:
            raise RuntimeError('Invalid neural extraction output shape')
        result={'method':args.method,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest(),
                'model_native_payload_bits':args.native_payload_bits,'tested_payload_bits':wm.size,
                'host':Path(args.host).name,'watermark':Path(args.watermark).name,
                'psnr':psnr(host,y),'ssim':ssim(host,y),
                'clean_ber':ber(wm,r),'clean_nc':nc(wm,r),
                'embedding_seconds_including_subprocess':timings[0],
                'extraction_seconds_including_subprocess':timings[1],
                'fairness_warning':'Subprocess timing is NOT directly comparable to in-process embedding FPS.',
                'system':platform.platform(),'status':'ran_with_supplied_checkpoint'}
        for attack,kw in [('jpeg40',{'quality':40}),('jpeg70',{'quality':70})]:
            attacked=apply_attack('jpeg',y,**kw)
            attack_file=temp/f'{attack}.png';output_file=temp/f'{attack}_decoded.png'
            cv2.imwrite(str(attack_file),attacked)
            cmd=[sys.executable,str(runner),'extract','--input',str(attack_file),
                 '--output',str(output_file),'--checkpoint',str(checkpoint),'--key',args.key,
                 '--watermark-size',str(args.payload_size)]
            subprocess.run(cmd,check=True,timeout=args.timeout,capture_output=True,text=True)
            recovered=cv2.imread(str(output_file),cv2.IMREAD_UNCHANGED)
            if recovered is None or recovered.shape[:2]!=wm.shape:
                raise RuntimeError(f'Invalid {attack} output')
            result[f'{attack}_ber']=ber(wm,recovered)
            result[f'{attack}_nc']=nc(wm,recovered)
    (output/'neural_baseline.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True,help='Verified neural model name and version')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--runner',required=True)
    p.add_argument('--native-payload-bits',required=True,type=int)
    p.add_argument('--payload-size',default=64,type=int)
    p.add_argument('--host',default='data/hosts/classical/girl.bmp')
    p.add_argument('--watermark',default='data/watermarks/watermark_1.png')
    p.add_argument('--key',default='review-audit-2026-fixed-key')
    p.add_argument('--timeout',default=300,type=int)
    p.add_argument('--output',default='results/reviewer_suite/neural_external')
    run(p.parse_args())
