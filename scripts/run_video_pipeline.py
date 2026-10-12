#!/usr/bin/env python3
"""Actual decoded/encoded video evaluation of the unchanged Green-Quad method."""
from __future__ import annotations
import argparse,csv,json,math,time,sys
from pathlib import Path
import cv2
import numpy as np
from qrwatermark.core.factory import build_method
from qrwatermark.utils.watermark import prepare_binary_watermark
from qrwatermark.evaluation.metrics import ber,nc,psnr,ssim


def writer_for(path: Path, codec: str, fps:float, size:tuple[int,int]):
    if len(codec)!=4:raise ValueError('codec must be a four-character FourCC')
    path.parent.mkdir(parents=True,exist_ok=True)
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*codec),fps,size)
    if not w.isOpened():raise RuntimeError(f'OpenCV cannot create encoded output {path} with codec {codec}')
    return w


def summarize(vals:list[float]):
    a=np.asarray(vals,dtype=float)
    if len(a)==0:return None
    return dict(mean=float(np.mean(a)),median=float(np.median(a)),p95=float(np.percentile(a,95)),min=float(np.min(a)),max=float(np.max(a)))


def run(args):
    inp=Path(args.input_video); out=Path(args.output_video); dest=Path(args.results_dir)
    if not inp.is_file():raise FileNotFoundError(inp)
    if inp.resolve()==out.resolve():raise ValueError('Input and output videos must be distinct')
    if args.max_frames<1:raise ValueError('max-frames must be positive')
    dest.mkdir(parents=True,exist_ok=True)
    summary={'completed':False,'codec':args.codec,'input':str(inp),'output':str(out),'watermark_bits':4096,
             'period':66,'pipeline_includes_decode_encode':True,'failure':None}
    cap=None;writer=None;cap_ref=None;frames=[];processed=[]
    try:
        root=Path(__file__).resolve().parents[1]
        wm_path=root/'data/watermarks/watermark_1.png'
        wm=prepare_binary_watermark(wm_path,size=64,background=255)
        black=int(np.count_nonzero(wm==0));white=int(np.count_nonzero(wm==255))
        if not black or not white:raise RuntimeError(f'Constant watermark after alpha compositing: {wm_path}')
        summary['watermark_bit_counts']={'black':black,'white':white}
        summary['watermark_preprocessing']='RGBA on white background; threshold 128'
        if not cv2.imwrite(str(dest/'expected_watermark.png'),wm):raise RuntimeError('Cannot save reference watermark')
        key=args.key.encode('utf8');method=build_method('proposed')
        cap=cv2.VideoCapture(str(inp))
        if not cap.isOpened():raise RuntimeError(f'Could not decode input video: {inp}')
        source_fps=float(cap.get(cv2.CAP_PROP_FPS))
        fps=(source_fps if math.isfinite(source_fps) and source_fps>0 else args.fallback_fps)
        summary['source_fps']=fps
        start=time.perf_counter()
        embed_times=[]; extract_times=[]; decode_times=[];encode_times=[]
        expected_shape=None
        for idx in range(args.max_frames):
            td=time.perf_counter(); ok,frame=cap.read();decode_times.append((time.perf_counter()-td)*1e3)
            if not ok:break
            if frame.ndim!=3 or frame.shape[2]!=3 or frame.dtype!=np.uint8:raise RuntimeError('Unsupported input frame')
            if (frame.shape[0]//4)*(frame.shape[1]//4)*4<4096:
                raise RuntimeError(f'Frame too small for 4096 bits: {frame.shape}')
            shape=frame.shape
            if expected_shape is None:
                expected_shape=shape
                writer=writer_for(out,args.codec,fps,(shape[1],shape[0]))
            elif shape!=expected_shape:raise RuntimeError('Input video changes resolution mid-stream')
            te=time.perf_counter(); embedded=method.embed(frame,wm,key=key)
            embed_ms=(time.perf_counter()-te)*1e3;embed_times.append(embed_ms)
            if embedded.side_info is not None:raise AssertionError('Embedding unexpectedly uses side information')
            tx=time.perf_counter(); raw=method.extract(embedded.image,key=key,watermark_shape=(64,64)).watermark
            extract_ms=(time.perf_counter()-tx)*1e3;extract_times.append(extract_ms)
            b=ber(wm,raw)
            if b!=0:raise AssertionError(f'Clean-watermarked frame {idx} did not decode exactly')
            tw=time.perf_counter();writer.write(embedded.image)
            encode_times.append((time.perf_counter()-tw)*1e3)
            frames.append(dict(frame=idx,embed_ms=embed_ms,extract_ms=extract_ms,clean_ber=float(b),
                               clean_nc=float(nc(wm,raw)),clean_psnr=float(psnr(frame,embedded.image)),
                               clean_ssim=float(ssim(frame,embedded.image))))
        if writer:writer.release();writer=None
        elapsed_embed=time.perf_counter()-start
        if not frames:raise RuntimeError('No input video frames decoded')
        summary['input_frames_processed']=len(frames);summary['decode_embed_encode_seconds']=elapsed_embed
        summary['decode_embed_encode_fps']=len(frames)/elapsed_embed
        if cap:cap.release();cap=None
        cap_out=cv2.VideoCapture(str(out))
        if not cap_out.isOpened():raise RuntimeError('Encoded video cannot be reopened')
        cap_ref=cv2.VideoCapture(str(inp))
        if not cap_ref.isOpened(): raise RuntimeError('Cannot reopen original video for post-codec fidelity')
        encoded_start=time.perf_counter()
        try:
            for idx in range(len(frames)):
                ok,frame=cap_out.read()
                if not ok:break
                if frame.shape!=expected_shape:raise RuntimeError('Codec changed frame dimensions')
                ref_ok,original_frame=cap_ref.read()
                if not ref_ok or original_frame.shape!=frame.shape:
                    raise RuntimeError('Could not align original and encoded frame for fidelity metrics')
                tx=time.perf_counter()
                recovered=method.extract(frame,key=key,watermark_shape=(64,64)).watermark
                ms=(time.perf_counter()-tx)*1e3
                processed.append(dict(frame=idx,encoded_extract_ms=ms,encoded_nc=float(nc(wm,recovered)),
                                      encoded_ber=float(ber(wm,recovered)),
                                      encoded_psnr=float(psnr(original_frame,frame)),
                                      encoded_ssim=float(ssim(original_frame,frame))))
            ok,extra=cap_out.read()
            if ok:raise RuntimeError('Encoded video has unexpected extra frames')
        finally:
            cap_out.release()
            if cap_ref:cap_ref.release();cap_ref=None
        elapsed_verify=time.perf_counter()-encoded_start
        if len(processed)!=len(frames):raise RuntimeError(f'Encoded video has {len(processed)} of {len(frames)} frames')
        total_elapsed=time.perf_counter()-start
        summary.update(decoded_output_frames=len(processed),output_redecode_extract_seconds=elapsed_verify,
                       total_e2e_seconds=total_elapsed,total_e2e_fps=len(frames)/total_elapsed,
                       embed_ms=summarize(embed_times),extract_ms=summarize(extract_times),
                       embed_extract_ms=summarize([x+y for x,y in zip(embed_times,extract_times)]),
                       encode_write_ms=summarize(encode_times),decode_read_ms=summarize(decode_times[:len(frames)]),
                       clean_psnr=summarize([r['clean_psnr'] for r in frames]),
                       clean_ssim=summarize([r['clean_ssim'] for r in frames]),
                       encoded_ber=summarize([r['encoded_ber'] for r in processed]),
                       encoded_nc=summarize([r['encoded_nc'] for r in processed]),
                       encoded_psnr=summarize([r['encoded_psnr'] for r in processed]),
                       encoded_ssim=summarize([r['encoded_ssim'] for r in processed]),completed=True)
        for name,rows in [('video_clean_frames.csv',frames),('video_encoded_frames.csv',processed)]:
            with (dest/name).open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return summary
    except Exception as e:
        summary['failure']=f'{type(e).__name__}: {e}'
        raise
    finally:
        if writer:writer.release()
        if cap:cap.release()
        if cap_ref:cap_ref.release()
        (dest/'video_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Encoded-video watermark evaluation; 4096-bit Green-Quad only')
    p.add_argument('--input-video',required=True,type=Path)
    p.add_argument('--output-video',required=True,type=Path)
    p.add_argument('--results-dir',required=True,type=Path)
    p.add_argument('--codec',default='mp4v',help='FourCC codec, e.g. mp4v or FFV1 (FFV1 use .mkv)')
    p.add_argument('--max-frames',type=int,default=300)
    p.add_argument('--fallback-fps',type=float,default=30.0)
    p.add_argument('--key',default='green-quad-video-v1')
    args=p.parse_args(argv)
    if args.fallback_fps<=0:p.error('fallback-fps must be >0')
    return args

if __name__=='__main__':
    print(json.dumps(run(parse_args()),indent=2))
