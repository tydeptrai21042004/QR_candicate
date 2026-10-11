# Kaggle ONE CELL: only proposed Green-Quad T=66 + internal ablations.
# Enable Internet for git clone, run as single notebook cell.
# Set QR_SMOKE_ONLY=1 for a fast correctness check.
import os,sys,subprocess,time,threading,zipfile,shutil,json,platform
from pathlib import Path

WORK=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
REPO=Path(os.environ.get('QR_LOCAL_REPO',str(WORK/'QR_candicate'))).resolve()
SRC='https://github.com/tydeptrai21042004/QR_candicate.git'
SMOKE=os.environ.get('QR_SMOKE_ONLY','0')=='1'
RESULTS=WORK/('GREEN_QUAD_SMOKE' if SMOKE else 'GREEN_QUAD_FULL')
ARCHIVE=WORK/('GREEN_QUAD_SMOKE.zip' if SMOKE else 'GREEN_QUAD_FULL_RESULTS.zip')
LOGS=RESULTS/'logs'
LOGS.mkdir(parents=True,exist_ok=True)
DEADLINE=time.monotonic()+11*3600+40*60
LOCK=threading.Lock()
STOP=threading.Event()

def checkpoint():
    with LOCK:
        temp=ARCHIVE.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for path in RESULTS.rglob('*'):
                if path.is_file():z.write(path,path.relative_to(WORK))
        temp.replace(ARCHIVE)

def periodic():
    while not STOP.wait(180):
        try:checkpoint()
        except Exception as exc:print('Checkpoint warning',exc,flush=True)
threading.Thread(target=periodic,daemon=True).start()

def task(name,cmd,timeout):
    if time.monotonic()>DEADLINE-180:raise TimeoutError('time budget exhausted')
    with (LOGS/f'{name}.log').open('w') as out:
        subprocess.run(cmd,cwd=REPO,stdout=out,stderr=subprocess.STDOUT,check=True,
                       timeout=max(1,min(timeout,int(DEADLINE-time.monotonic()-180))),
                       env={**os.environ,'PYTHONPATH':str(REPO/'src'),
                            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    checkpoint()

try:
    if not (REPO/'src'/'qrwatermark'/'proposed'/'green_quad_min_energy.py').exists():
        if REPO.exists() and any(REPO.iterdir()):
            if not (REPO/'.git').exists():raise RuntimeError('Nonempty repository destination is not a git checkout')
            subprocess.run(['git','-C',str(REPO),'fetch','--depth','1','origin','main'],check=True)
            subprocess.run(['git','-C',str(REPO),'reset','--hard','FETCH_HEAD'],check=True)
        else:subprocess.run(['git','clone','--depth','1',SRC,str(REPO)],check=True)
    subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(REPO),'pytest'],check=True)
    # Fail closed if stale V7/V8/V9 source is still in the checkout.
    old=[p.name for p in (REPO/'src/qrwatermark/proposed').glob('integer_convolution*.py')]
    if old:raise RuntimeError(f'Old proposals still present: {old}. Apply single-proposal patch first.')
    task('tests',[sys.executable,'-m','pytest','-q'],1200)
    task('ablation',[sys.executable,'scripts/run_green_quad_ablation.py',
         '--mode','quick' if SMOKE else 'full','--repeats','20' if SMOKE else '250',
         '--include-greedy-control','--output',str(RESULTS/'ablation')],42000)
    (RESULTS/'status.json').write_text(json.dumps({'success':True,'mode':'quick' if SMOKE else 'full','platform':platform.platform()},indent=2))
finally:
    STOP.set()
    checkpoint()
    print('Results:',ARCHIVE,flush=True)
