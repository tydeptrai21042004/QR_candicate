# Paste this entire file into ONE Kaggle notebook cell after pushing the patch
# to GitHub main. Turn Internet ON. CPU is sufficient.
# Attach a Kaggle dataset containing >=6 genuinely independent extra images.
# Change the NEXT line to the attached dataset folder:
EXTRA_REAL_HOSTS = '/kaggle/input/additional-real-host-images'

import os,sys,subprocess,zipfile,json,threading,time
from pathlib import Path

root=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
repo=root/'QR_candicate'
url='https://github.com/tydeptrai21042004/QR_candicate.git'
if (repo/'.git').exists():
    subprocess.run(['git','-C',str(repo),'fetch','--depth','1','origin','main'],check=True)
    subprocess.run(['git','-C',str(repo),'reset','--hard','FETCH_HEAD'],check=True)
else:
    subprocess.run(['git','clone','--depth','1',url,str(repo)],check=True)
revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','--short','HEAD'],text=True).strip()
for f in ['scripts/prepare_reviewer_hosts.py','src/qrwatermark/evaluation/review_statistics.py',
          'tests/test_evaluation_audit_v2.py']:
    if not (repo/f).is_file():
        raise RuntimeError('GitHub is missing the audit-v2 patch. Upload patch files to main first.')
extra=Path(os.getenv('QR_EXTRA_HOSTS_DIR',EXTRA_REAL_HOSTS))
if not extra.is_dir():
    raise FileNotFoundError('Attach at least 6 independent external real host images, then '
                            'set EXTRA_REAL_HOSTS to the Kaggle input directory: '+str(extra))
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(repo)+'[test]'],check=True)
env={**os.environ,'PYTHONPATH':str(repo/'src'), 'OMP_NUM_THREADS':'1',
     'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1','MPLBACKEND':'Agg'}
def execute(args):
    print('Running:', ' '.join(map(str,args)),flush=True)
    subprocess.run(args,cwd=repo,env=env,check=True)

execute([sys.executable,'-m','pytest','-q','tests/test_evaluation_audit_v2.py',
         'tests/test_reviewer_suite.py'])
host_dir=root/'qr_hosts_12'
execute([sys.executable,'scripts/prepare_reviewer_hosts.py',
         '--input',str(repo/'data/hosts/classical'),
         '--input',str(extra),'--output',str(host_dir)])
result=root/f'QR_REVIEWER_AUDIT_V2_{revision}'
result.mkdir(parents=True,exist_ok=True)
archive=root/(result.name+'.zip')
lock=threading.Lock();stop=threading.Event()
def checkpoint():
    with lock:
        tmp=archive.with_suffix('.zip.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
            for file in sorted(result.rglob('*')):
                if file.is_file():z.write(file,file.relative_to(root))
            for filename in ('source_manifest.csv','source_manifest.json'):
                f=host_dir/filename
                if f.exists():z.write(f,f.relative_to(root))
        tmp.replace(archive)
def checkpoints():
    while not stop.wait(180):
        try:checkpoint();print('Saved:',archive,flush=True)
        except Exception as exc:print('Checkpoint warning:',exc,flush=True)
t=threading.Thread(target=checkpoints,daemon=True);t.start()
try:
    for section in ('security','baselines'):
        execute([sys.executable,'scripts/run_reviewer_suite.py','--mode','full',
                 '--sections',section,'--hosts-dir',str(host_dir),
                 '--security-wrong-keys','64', '--output',str(result),'--fail-fast'])
        checkpoint()
finally:
    stop.set();t.join(timeout=5);checkpoint()
print('RESULT ARCHIVE:',archive)
try:
    from IPython.display import display,FileLink
    display(FileLink(str(archive)))
except ImportError:pass
