# Kaggle: paste the entire block into ONE Python cell. Enable Internet.
# This uses the corrected files committed to GitHub, without monkeypatching.
import os, subprocess, sys
from pathlib import Path

repo = Path('/kaggle/working/QR_candicate')
url = 'https://github.com/tydeptrai21042004/QR_candicate.git'
if (repo / '.git').exists():
    subprocess.run(['git','-C',str(repo),'fetch','--depth','1','origin','main'], check=True)
    subprocess.run(['git','-C',str(repo),'reset','--hard','FETCH_HEAD'], check=True)
elif repo.exists() and any(repo.iterdir()):
    raise RuntimeError(f'{repo} exists but is not a GitHub clone. Rename it and retry.')
else:
    subprocess.run(['git','clone','--depth','1',url,str(repo)],check=True)

runner = repo / 'kaggle' / 'QR_KAGGLE_GITHUB_ONE_CELL.py'
if not runner.is_file():
    raise RuntimeError('The corrected runner has not been committed to GitHub main. Upload the provided patch first.')
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(repo),'pytest'],check=True)
env={**os.environ,'QR_LOCAL_REPO':str(repo), 'QR_SMOKE_ONLY':'0'}
subprocess.run([sys.executable,str(runner)],cwd=repo,env=env,check=True)
print('Kaggle results: /kaggle/working/QR_ALL_COMPLETED_RESULTS.zip')
