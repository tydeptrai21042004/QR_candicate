# Paste this entire file into one Kaggle notebook Python cell AFTER the patch
# has been committed to the GitHub main branch. Kaggle Internet must be ON.
import json, sys, subprocess
from pathlib import Path
from IPython.display import display,FileLink

work=Path('/kaggle/working')
repo=work/'QR_candicate'
url='https://github.com/tydeptrai21042004/QR_candicate.git'
if (repo/'.git').is_dir():
    subprocess.run(['git','-C',str(repo),'fetch','--depth','1','origin','main'],check=True)
    subprocess.run(['git','-C',str(repo),'reset','--hard','FETCH_HEAD'],check=True)
elif not repo.exists():
    subprocess.run(['git','clone','--depth','1','--branch','main',url,str(repo)],check=True)
else:
    raise RuntimeError(f'{repo} exists but is not a Git checkout')
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
print('GREEN-QUAD COMMIT:',commit,flush=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(repo)+'[test]'],check=True)
optional=subprocess.run([sys.executable,'-m','pip','install','-q','imagecorruptions'],check=False)
if optional.returncode:print('WARNING: imagecorruptions is unavailable; official attacks requiring it may fail.')
runner=repo/'scripts/run_existing_hosts_mm_video.py'
if not runner.is_file():
    raise RuntimeError('The code patch has not been committed to GitHub. Add the patched files first.')
assert (repo/'data/hosts/classical').is_dir(), 'Existing 6-host dataset missing from GitHub checkout'
output=work/'MMWATERMARK_VIDEO_RESULTS'
cmd=[sys.executable,'-u',str(runner),'--output',str(output),
     '--min-images','6','--max-images','6',
     '--severities','1','2','3','4','5',
     '--video-frames','300']
# To use your own real video, append: cmd += ['--video','/kaggle/input/.../clip.mp4']
finished=subprocess.run(cmd,cwd=repo,check=False)
status_file=output/'execution_status.json'
if status_file.exists():
    status=json.loads(status_file.read_text())
    for name,item in status['benchmarks'].items():
        print(name,'=>',item['status'],item.get('error',''))
    print('ALL COMPLETED:',status['completed'])
    display(FileLink(str(status_file)))
for file in [output/'MMWATERMARK_VIDEO_RESULTS.zip',output/'video/FFV1/watermarked.mkv',output/'video/mp4v/watermarked.mp4']:
    if file.is_file():display(FileLink(str(file)))
print('RUNNER RETURN CODE:',finished.returncode)
