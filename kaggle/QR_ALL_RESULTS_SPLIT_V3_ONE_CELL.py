# Paste this entire file into ONE Kaggle code cell. Enable Internet.
# Optional: attach >=6 independent REAL host images and set EXTRA_REAL_HOSTS.
# No data fabrication. In absence of extra hosts, results are SIX-HOST EXPLORATORY.
import os,sys,subprocess
from pathlib import Path

GITHUB='https://github.com/tydeptrai21042004/QR_candicate.git'
WORK=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
REPO=WORK/'QR_candicate'
EXTRA_REAL_HOSTS=os.getenv('QR_EXTRA_HOSTS_DIR','')  # e.g. /kaggle/input/my-real-images
MODE=os.getenv('QR_EXPERIMENT_MODE','full')           # 'smoke' or 'full'
ONLY=os.getenv('QR_EXPERIMENT_ONLY','all')             # 'all', 'ablation', 'hyperparameters', 'review', 'baselines', 'tests'
MAX_HOURS=os.getenv('QR_MAX_HOURS','11.58')           # leaves time for final ZIP

if (REPO/'.git').is_dir():
    subprocess.run(['git','-C',str(REPO),'fetch','--depth','1','origin','main'],check=True)
    subprocess.run(['git','-C',str(REPO),'reset','--hard','FETCH_HEAD'],check=True)
elif not REPO.exists():
    subprocess.run(['git','clone','--depth','1','--branch','main',GITHUB,str(REPO)],check=True)
else:
    raise RuntimeError('Repository path already exists without .git: '+str(REPO))
required=['scripts/run_qr_complete_suite.py','scripts/run_green_quad_hyperparameters.py',
          'scripts/run_green_quad_ablation.py','tests/test_qr_complete_runner_v3.py',
          'scripts/run_reviewer_suite.py','scripts/prepare_reviewer_hosts.py']
missing=[x for x in required if not (REPO/x).is_file()]
if missing:raise RuntimeError('Push the V3 patch to GitHub main first: '+str(missing))
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(REPO)+'[test]'],check=True)
cmd=[sys.executable,'scripts/run_qr_complete_suite.py','--mode',MODE,'--only',ONLY,
     '--max-hours',MAX_HOURS,'--out-root',str(WORK)]
if EXTRA_REAL_HOSTS:cmd+=['--extra-hosts-dir',EXTRA_REAL_HOSTS]
# Collect ALL completed outputs even if individual evaluation jobs fail.
res=subprocess.run(cmd,cwd=REPO)
print('Experiment runner exit code:',res.returncode)
from IPython.display import display,FileLink
zips=sorted(WORK.glob(f'QR_ALL_RESULTS_V3_*_{MODE.upper()}.zip'),key=lambda x:x.stat().st_mtime,reverse=True)
if zips:
    print('Latest consolidated results:',zips[0]);display(FileLink(str(zips[0])))
else:
    print('No result ZIP created; check the runner error above.')
