# Kaggle ONE CELL bootstrap for single Green-Quad T66 experiment.
import os,sys,subprocess
from pathlib import Path
repo=Path('/kaggle/working/QR_candicate')
url='https://github.com/tydeptrai21042004/QR_candicate.git'
if (repo/'.git').is_dir():
    subprocess.run(['git','-C',str(repo),'fetch','--depth','1','origin','main'],check=True)
    subprocess.run(['git','-C',str(repo),'reset','--hard','FETCH_HEAD'],check=True)
else:
    subprocess.run(['git','clone','--depth','1',url,str(repo)],check=True)
subprocess.run([sys.executable,str(repo/'kaggle'/'QR_KAGGLE_GITHUB_ONE_CELL.py')],
               cwd=repo,env={**os.environ,'QR_LOCAL_REPO':str(repo)},check=True)
