# Kaggle single Python cell: clone updated GitHub code, verify fixes, run both
# watermarks through MMWatermark official perturbations and FFV1/mp4v video.
# Push the accompanying final patch to GitHub before executing this cell.
import json
import os
from pathlib import Path
import subprocess
import sys
from IPython.display import FileLink, display

ROOT = Path('/kaggle/working')
REPO = ROOT / 'QR_candicate'
GITHUB = 'https://github.com/tydeptrai21042004/QR_candicate.git'
RUN_FULL_EXPERIMENT_SUITE = False  # Set True to rerun the expensive main 10-job suite
STRICT_MMWATERMARK = True         # Report failure if any official attack is unavailable
VIDEO_FRAMES = 300
USER_VIDEO = ''                    # Set to an existing video path to avoid sample download

ENV = {**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MPLBACKEND': 'Agg'}

if (REPO / '.git').exists():
    subprocess.run(['git', '-C', str(REPO), 'fetch', '--depth', '1', 'origin', 'main'], check=True)
    subprocess.run(['git', '-C', str(REPO), 'reset', '--hard', 'FETCH_HEAD'], check=True)
elif not REPO.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '-b', 'main', GITHUB, str(REPO)], check=True)
else:
    raise RuntimeError('Existing non-git folder conflicts with repository checkout')
commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
print('Commit:', commit)
for relative in ('scripts/run_mmwatermark_official.py', 'scripts/run_existing_hosts_mm_video.py',
                 'scripts/run_video_pipeline.py', 'data/watermarks/watermark_1.png',
                 'data/watermarks/watermark_2.png'):
    if not (REPO / relative).is_file():
        raise FileNotFoundError('Missing required repository file: ' + relative)
if '--watermark-path' not in (REPO / 'scripts/run_video_pipeline.py').read_text():
    raise RuntimeError('Final two-watermark patch has not been pushed to GitHub')
if 'adapt_skimage_gaussian' not in (REPO / 'scripts/run_mmwatermark_official.py').read_text():
    raise RuntimeError('Final MMWatermark fix has not been pushed to GitHub')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(REPO) + '[test]'], check=True)
# Some official perturbations require optional scientific-image packages.
opt = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'imagecorruptions', 'wand'],
                     check=False)
if opt.returncode:
    print('WARNING: Optional upstream dependencies could not be installed.')

OUT = ROOT / f'QR_FINAL_{commit[:8]}'
OUT.mkdir(parents=True, exist_ok=True)
results = {}

def job(name, cmd):
    logfile = OUT / (name + '.log')
    print('\n' + '=' * 70 + '\nRUN: ' + name, flush=True)
    with logfile.open('w', encoding='utf8') as f:
        proc = subprocess.Popen([str(x) for x in cmd], cwd=REPO, env=ENV,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors='replace')
        for line in proc.stdout:
            print(line, end='', flush=True)
            f.write(line)
        ret = proc.wait()
    results[name] = {'status': 'PASS' if ret == 0 else 'FAILED',
                     'return_code': ret, 'log': str(logfile)}
    print('RESULT:', results[name]['status'], flush=True)

job('tests', [sys.executable, '-m', 'pytest', '-q',
    'tests/test_green_quad_min_energy.py', 'tests/test_final_single_proposal.py',
    'tests/test_baselines_smoke.py', 'tests/test_paper_baselines.py',
    'tests/test_reviewer_suite.py', 'tests/test_evaluation_audit_v2.py',
    'tests/test_qr_complete_runner_v3.py', 'tests/test_mmwatermark_official_integration.py',
    'tests/test_existing_hosts_mm_video_fix.py', 'tests/test_video_pipeline_integration.py'])

cmd = [sys.executable, '-u', 'scripts/run_existing_hosts_mm_video.py',
       '--output', str(OUT / 'MM_VIDEO_2WM'), '--min-images', '6', '--max-images', '6',
       '--severities', '1', '2', '3', '4', '5', '--video-frames', str(VIDEO_FRAMES)]
if STRICT_MMWATERMARK:
    cmd.append('--strict-mm')
if USER_VIDEO:
    cmd.extend(['--video', USER_VIDEO])
job('mm_video_both_watermarks', cmd)
status_path = OUT / 'MM_VIDEO_2WM' / 'execution_status.json'
if status_path.is_file():
    external = json.loads(status_path.read_text())
    results['external_statuses'] = {k: v['status'] for k, v in external.get('benchmarks', {}).items()}
    print('External status:', results['external_statuses'])

if RUN_FULL_EXPERIMENT_SUITE:
    job('full_experiments', [sys.executable, '-u', 'scripts/run_qr_complete_suite.py',
        '--mode', 'full', '--only', 'all', '--out-root', str(OUT),
        '--max-hours', '10.5', '--reserve-minutes', '8', '--repeats', '250',
        '--security-wrong-keys', '64', '--checkpoint-seconds', '240'])

status_file = OUT / 'FINAL_EXECUTION_STATUS.json'
status_file.write_text(json.dumps({'git_commit': commit, 'results': results}, indent=2))
for zipfile in sorted(OUT.glob('**/*.zip')):
    print('RESULT ZIP:', zipfile)
    display(FileLink(str(zipfile)))
display(FileLink(str(status_file)))
print('Check statuses: MMWatermark strict coverage and the four video jobs must pass.')
