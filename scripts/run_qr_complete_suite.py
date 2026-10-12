"""Crash-resilient, restartable Green-Quad reviewer evaluation orchestrator.

Full: tests, two distinct studies, reviewer security/attacks/optimality/capacity/
runtime, two paper comparisons, and locked-validation matched baselines.
A study is considered complete ONLY if subprocess succeeds AND expected files exist.
The script stores independent outputs, logs, durable status and ZIP checkpoints.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, shutil, signal, subprocess, sys, threading, time, zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST_EXTS = {'.bmp','.png','.jpg','.jpeg','.tif','.tiff'}

@dataclass(frozen=True)
class Job:
    name: str
    group: str
    cmd: tuple[str, ...]
    required: tuple[str, ...] = ()


def atomic_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf8')
    os.replace(tmp,path)


def repo_revision():
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'local-unversioned'


def corpus_fingerprint(folder):
    hashes=hashlib.sha256()
    files=sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in HOST_EXTS),key=lambda p:p.name)
    for path in files:
        hashes.update(path.name.encode())
        with path.open('rb') as f:
            while True:
                b=f.read(1<<20)
                if not b:break
                hashes.update(b)
    return hashes.hexdigest()[:12],len(files)


def make_jobs(args, run_dir, hosts):
    p=sys.executable
    suffix=['--hosts-dir',str(hosts),'--min-hosts',str(args.min_hosts)]
    mode='quick' if args.mode=='smoke' else 'full'
    rev_mode='smoke' if args.mode=='smoke' else 'full'
    repeats=str(3 if args.mode=='smoke' else args.repeats)
    study_common=['--mode',mode,'--repeats',repeats,*suffix,'--partition','all']
    reviewer=['scripts/run_reviewer_suite.py','--mode',rev_mode,
        '--hosts-dir',str(hosts),'--min-hosts',str(args.min_hosts),
        '--security-wrong-keys',str(args.security_wrong_keys),
        '--output',str(run_dir/'reviewer_full'),'--fail-fast']
    jobs=[
        Job('maintained_tests','tests',(p,'-m','pytest','-q',
                'tests/test_green_quad_min_energy.py','tests/test_final_single_proposal.py',
                'tests/test_baselines_smoke.py','tests/test_paper_baselines.py',
                'tests/test_reviewer_suite.py','tests/test_evaluation_audit_v2.py',
                'tests/test_qr_complete_runner_v3.py',
                '--junitxml',str(run_dir/'maintained_tests.xml')),
                ('maintained_tests.xml',)),
        Job('ablation','ablation',(p,'scripts/run_green_quad_ablation.py',
                *study_common,'--include-greedy-control',
                '--output',str(run_dir/'ablation')),
                ('ablation/summary.csv','ablation/by_attack.csv',
                 'ablation/ablation_deltas.csv','ablation/attack_rows.csv',
                 'ablation/runtime.csv','ablation/multikey_clean.csv','ablation/study_protocol.json')),
        Job('hyperparameter_analysis','hyperparameters',
                (p,'scripts/run_green_quad_hyperparameters.py',*study_common,
                 '--output',str(run_dir/'hyperparameter_analysis')),
                ('hyperparameter_analysis/summary.csv','hyperparameter_analysis/by_attack.csv',
                 'hyperparameter_analysis/hyperparameter_selection.json',
                 'hyperparameter_analysis/attack_rows.csv',
                 'hyperparameter_analysis/runtime.csv','hyperparameter_analysis/study_protocol.json',
                 'hyperparameter_analysis/analysis_summary.json')),
    ]
    for section, outputs in (
            ('security',('security/report.json','security/roc.csv','security/raw.csv')),
            ('attacks',('attacks/raw.csv','attacks/per_attack_ci.csv')),
            ('optimality',('optimality/integer_projection.csv',)),
            ('capacity',('capacity/capacity.csv',)),
            ('runtime',('runtime/runtime.csv','runtime/environment.json'))):
        jobs.append(Job('reviewer_'+section,'review',
                  tuple([p,*reviewer,'--sections',section]),
                  tuple('reviewer_full/'+x for x in outputs)))
    if args.mode=='full':
        for label,config in [('paper_baselines_256bit',
                'configs/experiments/qim_control_comparison.yaml'),
                             ('paper_baselines_standard','configs/experiments/main_comparison.yaml')]:
            jobs.append(Job(label,'baselines',
                 (p,'scripts/run_main_comparison.py','--config',config,
                  '--key','review-audit-2026-fixed-key',
                  '--run-dir',str(run_dir/label),*suffix),
                 (label+'/metrics.csv',label+'/summary.csv',
                  label+'/psnr_audit.csv',label+'/unrestricted/paired_host_ci.csv')))
    # Per the chosen study protocol, matched-PSNR calibration is optional;
    # do not run it by default or use it as an operating-point acceptance gate.
    if getattr(args, 'include_matched_baselines', False):
        jobs.append(Job('reviewer_baselines_matched','baselines',
                 tuple([p,*reviewer,'--sections','baselines']),
                 ('reviewer_full/baselines/metrics_with_eligibility.csv',
                  'reviewer_full/baselines/psnr_coverage.csv',
                  'reviewer_full/baselines/matched_only/paired_host_ci.csv')))
    if args.neural_checkpoint or args.neural_runner:
        if not args.neural_checkpoint or not args.neural_runner:
            raise ValueError('Supply BOTH --neural-runner and --neural-checkpoint')
        jobs.append(Job('external_neural_baseline','neural',
            (p,'scripts/run_external_neural_baseline.py',
             '--method',args.neural_method,'--checkpoint',str(args.neural_checkpoint),
             '--runner',str(args.neural_runner),'--native-payload-bits','4096',
             '--output',str(run_dir/'neural')),('neural/neural_baseline.json',)))
    return jobs


def accepted(job,base):
    if not all((base/rel).is_file() and (base/rel).stat().st_size>0
               for rel in job.required):
        return False
    if job.name == 'reviewer_runtime':
        # Upstream runtime catches per-method exceptions and exits 0. Never
        # report a complete runtime comparison containing hidden error rows.
        with (base/'reviewer_full/runtime/runtime.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        if not rows or any(row.get('error','').strip() for row in rows):
            return False
        expected={'green_quad_min_energy','direct_pixel_qim_control',
                  'su2014_qr','su2016_hessenberg','su2020_schur',
                  'chen2021_qqrd','nha2022_improved_qr'}
        if {r.get('method') for r in rows}!=expected:
            return False
    return True


def checkpoint(run_dir,zip_file,lock):
    """Use temporary archive and atomic rename; never publish truncated ZIP."""
    with lock:
        tmp=zip_file.with_suffix('.zip.tmp')
        try:
            with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=3,allowZip64=True) as archive:
                for file in sorted(run_dir.rglob('*')):
                    if file.is_file() and '.tmp' not in file.name:
                        archive.write(file,arcname=str(file.relative_to(run_dir.parent)))
            os.replace(tmp,zip_file)
        finally:
            tmp.unlink(missing_ok=True)


def stop_group(proc):
    try:
        if os.name=='posix':os.killpg(proc.pid,signal.SIGTERM)
        else:proc.terminate()
    except ProcessLookupError:pass
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            if os.name=='posix':os.killpg(proc.pid,signal.SIGKILL)
            else:proc.kill()
        except ProcessLookupError:pass
        proc.wait()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('smoke','full'),default='full')
    ap.add_argument('--hosts-dir',type=Path,default=None)
    ap.add_argument('--extra-hosts-dir',type=Path,default=None,
        help='Additional genuine independent scenes (input images normalized by prepare_reviewer_hosts.py)')
    ap.add_argument('--min-hosts',type=int,default=None)
    ap.add_argument('--out-root',type=Path,default=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else ROOT/'results')
    ap.add_argument('--max-hours',type=float,default=11.58)
    ap.add_argument('--reserve-minutes',type=float,default=8)
    ap.add_argument('--repeats',type=int,default=250)
    ap.add_argument('--security-wrong-keys',type=int,default=64)
    ap.add_argument('--include-matched-baselines',action='store_true',
        help='Optional separate locked-validation PSNR-matched baseline audit')
    ap.add_argument('--checkpoint-seconds',type=int,default=240)
    ap.add_argument('--only',choices=('all','ablation','hyperparameters','review','baselines','tests','neural'),default='all')
    ap.add_argument('--neural-runner',type=Path,default=None)
    ap.add_argument('--neural-checkpoint',type=Path,default=None)
    ap.add_argument('--neural-method',default='independent_pretrained_neural')
    args=ap.parse_args(argv)
    if args.max_hours<=0 or args.reserve_minutes<0 or args.repeats<3:
        ap.error('Positive --max-hours and --repeats>=3 required')
    root=args.out_root.resolve();root.mkdir(parents=True,exist_ok=True)
    extra=args.extra_hosts_dir
    if extra is not None:
        if not extra.is_dir():ap.error(f'Additional host corpus does not exist: {extra}')
        target=root/'QR_PREPARED_HOSTS_12'
        if target.exists():shutil.rmtree(target)  # Prevent stale mixed corpora.
        # Prepared data is checked for decoded-pixel duplication and shape.
        from prepare_reviewer_hosts import prepare
        prepare([ROOT/'data/hosts/classical',extra],target,minimum=12)
        host_dir=target
    else:
        host_dir=(args.hosts_dir or ROOT/'data/hosts/classical').resolve()
    if not host_dir.is_dir():ap.error(f'Host directory not found: {host_dir}')
    fingerprint,n_hosts=corpus_fingerprint(host_dir)
    args.min_hosts = args.min_hosts if args.min_hosts is not None else (12 if extra is not None else 6)
    if n_hosts<args.min_hosts:ap.error(f'Only {n_hosts} host images; require {args.min_hosts}')
    sha=repo_revision()
    run_dir=root/f'QR_ALL_RESULTS_V3_{sha[:8]}_{fingerprint}_{args.mode.upper()}'
    run_dir.mkdir(parents=True,exist_ok=True)
    zip_file=root/(run_dir.name+'.zip')
    lock=threading.Lock();stop=threading.Event()
    jobs=make_jobs(args,run_dir,host_dir)
    selected=[j for j in jobs if args.only=='all' or j.group==args.only]
    if not selected:ap.error(f'No tasks for group {args.only} in mode {args.mode}')
    source_id={'source_revision':sha,'dataset_sha':fingerprint,'mode':args.mode,
        'n_hosts':n_hosts,'min_hosts':args.min_hosts,
        'repeats':args.repeats,'security_wrong_keys':args.security_wrong_keys,
        'jobs':{j.name:list(j.cmd) for j in jobs}}
    status_file=run_dir/'job_status.json'
    if status_file.exists():
        old=json.loads(status_file.read_text(encoding='utf8'))
        if old.get('source_id')!=source_id:
            # Important: never reuse metrics from another command/corpus.
            raise RuntimeError('Existing results have different source or commands; use another result root.')
        status=old
    else:
        status={'source_id':source_id,'planned':[j.name for j in jobs],
                'completed':[],'failed':{},'timed_out':[], 'skipped':{},
                'running':None,'started_at_unix':time.time(),
                'exploratory_under_12':n_hosts<12,
                'optional_neural':'included' if args.neural_checkpoint else 'not supplied',
                'environment':{'platform':platform.platform(),'python':sys.version,
                               'repo':str(ROOT),'hosts':str(host_dir)}}
    status['running']=None
    status['last_update_unix']=time.time()
    atomic_json(status_file,status)
    def autosave():
        while not stop.wait(max(10,args.checkpoint_seconds)):
            try:
                checkpoint(run_dir,zip_file,lock)
                print('[checkpoint]',zip_file,flush=True)
            except Exception as exc:
                print('[checkpoint warning]',repr(exc),flush=True)
    bg=threading.Thread(target=autosave,daemon=True);bg.start()
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'NUMEXPR_NUM_THREADS':'1','MPLBACKEND':'Agg','PYTHONUNBUFFERED':'1'}
    deadline=time.monotonic()+args.max_hours*3600
    reserve=args.reserve_minutes*60
    try:
        for j in selected:
            if j.name in status['completed'] and accepted(j,run_dir):
                print('[resume:verified]',j.name,flush=True);continue
            if time.monotonic()>=deadline-reserve:
                print('[deadline] budget reserved to archive finished results',flush=True)
                status['timed_out']=[x.name for x in selected if x.name not in status['completed']]
                break
            (run_dir/'logs').mkdir(exist_ok=True)
            log=run_dir/'logs'/f'{j.name}.log'
            status['running']=j.name
            atomic_json(status_file,status)
            print('[running]',j.name,flush=True)
            start=time.monotonic();timedout=False
            with log.open('w',encoding='utf8') as output:
                proc=subprocess.Popen(list(j.cmd),cwd=ROOT,env=env,
                                      stdout=output,stderr=subprocess.STDOUT,
                                      start_new_session=(os.name=='posix'))
                try:
                    while proc.poll() is None:
                        if time.monotonic()>=deadline-reserve:
                            timedout=True;stop_group(proc);break
                        time.sleep(2)
                except KeyboardInterrupt:
                    stop_group(proc);timedout=True;raise
            elapsed=round(time.monotonic()-start,1)
            is_valid=proc.returncode==0 and accepted(j,run_dir) and not timedout
            if is_valid:
                if j.name not in status['completed']:status['completed'].append(j.name)
                status['failed'].pop(j.name,None)
                print('[completed]',j.name,elapsed,'sec',flush=True)
            elif timedout:
                status['timed_out'].append(j.name)
                print('[stopped at deadline]',j.name,flush=True)
            else:
                status['failed'][j.name]={'exit_code':proc.returncode,
                    'elapsed_seconds':elapsed,
                    'missing_outputs':[r for r in j.required if not (run_dir/r).is_file()],
                    'invalid_outputs':proc.returncode==0 and not accepted(j,run_dir)}
                print('[failed]',j.name,'exit',proc.returncode,flush=True)
            tail=log.read_text(encoding='utf8',errors='replace').splitlines()[-5:]
            for line in tail:print('  '+line,flush=True)
            status['running']=None;status['last_update_unix']=time.time()
            atomic_json(status_file,status)
            checkpoint(run_dir,zip_file,lock)
            if timedout:break
    except KeyboardInterrupt:
        print('[interrupted] saving all completed outputs',flush=True)
    finally:
        status['running']=None;status['last_update_unix']=time.time()
        status['missing_selected']=[j.name for j in selected if j.name not in status['completed']]
        status['all_selected_completed']=not status['missing_selected']
        if args.mode=='smoke':status['skipped']['paper_comparisons']='Not run in smoke; run --mode full'
        if n_hosts<12:status['skipped']['12_host_independence']='Only six provided hosts; exploratory outputs only'
        atomic_json(status_file,status)
        stop.set();bg.join(timeout=5)
        checkpoint(run_dir,zip_file,lock)
        print('\nRESULT ZIP:',zip_file,flush=True)
        print('COMPLETED:',status['completed'],'FAILED:',status['failed'],flush=True)
        print('INCOMPLETE:',status['missing_selected'],flush=True)
    return 0 if status['all_selected_completed'] else 2

if __name__=='__main__':
    raise SystemExit(main())
