# Kaggle: paste this entire file into ONE Python notebook cell. Internet ON.
# It clones the corrected GitHub repository, verifies committed dependencies,
# checks tests + clean decoding, then runs PSNR-matched paper
# baselines, same-strength proposal controls, v9 ablations, and runtime.
# Edit SMOKE_ONLY=True below to run a quick end-to-end smoke test.
import os, sys, json, time, shutil, zipfile, subprocess, threading, traceback, hashlib
from pathlib import Path
from datetime import datetime, timezone

REPO_URL = 'https://github.com/tydeptrai21042004/QR_candicate.git'
KEY = 'fixed-reproducibility-key-20261010'
SMOKE_ONLY = os.environ.get('QR_SMOKE_ONLY', '0') == '1'
MAX_SECONDS = 11*3600 + 40*60
RESERVE_SECONDS = 8*60
START = time.monotonic()
WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/mnt/data/QR_KAGGLE_LOCAL')
ROOT = Path(os.environ.get('QR_LOCAL_REPO', str(WORK/'QR_candicate'))).resolve()
OUT = WORK / ('QR_GITHUB_SMOKE_RESULTS' if SMOKE_ONLY else 'QR_GITHUB_BENCHMARK_RESULTS')
ARCHIVE = WORK / ('QR_GITHUB_SMOKE_RESULTS.zip' if SMOKE_ONLY else 'QR_ALL_COMPLETED_RESULTS.zip')
OUT.mkdir(parents=True,exist_ok=True)
LOGS=OUT/'logs'; LOGS.mkdir(exist_ok=True)
CFGS=OUT/'generated_configs'; CFGS.mkdir(exist_ok=True)
STATE=OUT/'run_state.json'
zip_lock=threading.Lock(); state_lock=threading.Lock(); stop=threading.Event()
completed=[]; failed=[]
STALE_RUN=False
fatal_error=None



def utc():
    return datetime.now(timezone.utc).isoformat()

def remaining():
    return MAX_SECONDS - (time.monotonic()-START)

def save_state():
    with state_lock:
        payload={'utc':utc(),'repo':str(ROOT),'smoke_only':SMOKE_ONLY,
                 'completed':completed,'failed':failed,'git_commit':globals().get('COMMIT',''),
                 'elapsed_seconds':round(time.monotonic()-START,2)}
        p=STATE.with_suffix('.tmp')
        p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        os.replace(p,STATE)

def checkpoint():
    with zip_lock:
        tmp=ARCHIVE.with_name(ARCHIVE.name+'.tmp')
        try:
            with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
                for p in sorted(OUT.rglob('*')):
                    if p.is_file() and not p.is_symlink():
                        z.write(p,str(p.relative_to(WORK)))
            os.replace(tmp,ARCHIVE)
            print('CHECKPOINT',ARCHIVE,'MB',round(ARCHIVE.stat().st_size/1e6,2),flush=True)
        except Exception as exc:
            print('WARNING checkpoint:',repr(exc),flush=True)
            if tmp.exists():tmp.unlink()

def watchdog():
    while not stop.wait(180):
        save_state();checkpoint()


def cmd(label, command, *, limit=1800, expected=None):
    if label in completed and (expected is None or Path(expected).exists()):
        print('RESUME: skipping',label,flush=True)
        return True
    if remaining()<=RESERVE_SECONDS:
        print('BUDGET: no new work',flush=True)
        return False
    timeout=max(1,int(min(limit,remaining()-RESERVE_SECONDS)))
    log=LOGS/(label+'.log')
    print('RUN',label,'timeout',timeout,'seconds',flush=True)
    try:
        with log.open('w',encoding='utf-8') as f:
            f.write(' '.join(map(str,command))+'\n');f.flush()
            subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                           check=True,timeout=timeout,
                           env={**os.environ,'PYTHONPATH':str(ROOT/'src'),
                                'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                                'MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
        if expected and not Path(expected).exists():
            raise FileNotFoundError(f'missing expected result: {expected}')
    except Exception as exc:
        msg={'job':label,'error':repr(exc),'log':str(log.relative_to(OUT))}
        failed.append(msg)
        print('FAILED',label,repr(exc),flush=True)
        try:
            lines=log.read_text(errors='replace').splitlines()
            print('\n'.join(lines[-24:]),flush=True)
        except Exception:pass
        save_state();checkpoint()
        return False
    completed.append(label)
    save_state();checkpoint()
    return True




def attacks25():
    # One clean plus 25 named, explicit attacked settings, fixed seeds.
    specs=[{'name':'clean'},
           {'name':'gaussian_blur','params':{'sigma':0.7}},
           {'name':'gaussian_blur','params':{'sigma':1.0}},
           {'name':'lowpass','params':{'kx':5,'ky':5}},
           {'name':'average','params':{'ksize':3}},
           {'name':'motion_blur','params':{'ksize':5}},
           {'name':'jpeg','params':{'quality':90}},
           {'name':'jpeg','params':{'quality':70}},
           {'name':'jpeg','params':{'quality':50}}]
    for v in (0.001,0.003):
        for s in (0,1,2):
            specs.append({'name':'gaussian_noise','params':{'variance':v},'seeds':[s]})
    for d in (0.02,0.05):
        for s in (0,1,2):
            specs.append({'name':'salt_pepper','params':{'density':d},'seeds':[s]})
    specs.extend([
        {'name':'scale_resample','params':{'scale':0.8}},
        {'name':'registered_rotation_resample','params':{'angle':5}},
        {'name':'rotation_unregistered','params':{'angle':2}},
        {'name':'translation','params':{'dx':2,'dy':2}},
        {'name':'crop_resize','params':{'fraction':0.05}},
    ])
    assert len(specs)==26
    return specs


def write_yaml(path, obj, yaml):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(obj,sort_keys=False),encoding='utf-8')


def join_csv(outglob, filename, pd):
    paths=sorted(OUT.glob(outglob))
    frames=[]
    for p in paths:
        try:
            if p.stat().st_size:
                d=pd.read_csv(p)
                if not d.empty:frames.append(d)
        except Exception as exc:
            failed.append({'csv':str(p),'error':repr(exc)})
    if frames:
        df=pd.concat(frames,ignore_index=True,sort=False)
        if 'PAPER_BASELINES' in filename:
            common=['method','host','watermark','attack','attack_params','seed']
            df=df.drop_duplicates(subset=[x for x in common if x in df],keep='first')
        df.to_csv(OUT/filename,index=False)
        return df
    return pd.DataFrame()


def summary(df, name):
    if df.empty:return
    attacked=df.loc[df['attack']!='clean']
    clean=df.loc[df['attack']=='clean']
    if not attacked.empty:
        g=attacked.groupby('method').agg(mean_nc=('nc','mean'),mean_ber=('ber','mean'),
             attacked_rows=('nc','count'))
        ps=clean.groupby('method').agg(min_clean_psnr=('embedding_psnr','min'),
              mean_clean_psnr=('embedding_psnr','mean'),max_clean_ber=('ber','max'))
        g.join(ps).to_csv(OUT/(name+'_SUMMARY.csv'))
        attacked.groupby(['method','attack'])[['nc','ber']].agg(['mean','count']).to_csv(OUT/(name+'_BY_ATTACK.csv'))
        print(name,'\n',g.join(ps).to_string(),flush=True)

try:
    if not os.environ.get('QR_LOCAL_REPO'):
        if (ROOT/'.git').exists():
            subprocess.run(['git', '-C', str(ROOT), 'fetch', '--depth', '1', 'origin', 'main'],
                           check=True, timeout=180)
            subprocess.run(['git', '-C', str(ROOT), 'reset', '--hard', 'FETCH_HEAD'],
                           check=True, timeout=60)
        else:
            if ROOT.exists() and any(ROOT.iterdir()):
                raise RuntimeError('Checkout path exists but is not a Git clone: ' + str(ROOT))
            subprocess.run(['git','clone','--depth','1',REPO_URL,str(ROOT)],
                           check=True,timeout=180)
    elif not (ROOT/'pyproject.toml').is_file():
        raise RuntimeError('QR_LOCAL_REPO is not a repository: ' + str(ROOT))
    os.chdir(ROOT)
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    commit=git.stdout.strip() or 'UNAVAILABLE_LOCAL_COPY'
    COMMIT=commit
    if STATE.exists():
        try:
            prev=json.loads(STATE.read_text(encoding='utf-8'))
            if prev.get('git_commit') == commit and prev.get('smoke_only') == SMOKE_ONLY:
                completed[:]=prev.get('completed', [])
                print('RESUMING',len(completed),'previous successful jobs',flush=True)
            else:
                STALE_RUN=True
                print('Revision/mode changed. Existing completed results are preserved.',flush=True)
                raise RuntimeError('Rename '+str(OUT)+' and retry to avoid mixing results from distinct commits')
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError('Cannot trust previous run_state.json') from exc
    (OUT/'git_commit.txt').write_text(commit+'\n')
    # This version contains NO local monkeypatch. GitHub must already be fixed.
    proposed = ROOT / 'src/qrwatermark/proposed'
    required = [proposed / 'block_carrier.py',
                proposed / 'integer_convolution_v7.py',
                proposed / 'integer_convolution_pair_v8.py',
                proposed / 'integer_convolution_quad_v9.py',
                proposed / 'method.py',
                ROOT / 'tests/test_recovered_block_carrier.py']
    missing = [str(f.relative_to(ROOT)) for f in required if not f.is_file()]
    if missing:
        raise RuntimeError('GitHub main is not repaired. Missing files: ' + repr(missing) +
                           '. Upload QR_GITHUB_CHANGED_FILES.zip to the repository and commit it first.')
    for name in ('integer_convolution_v7.py', 'integer_convolution_pair_v8.py',
                 'integer_convolution_quad_v9.py'):
        body = (proposed/name).read_text(encoding='utf-8')
        if 'from .block_carrier import _gather_blocks, keyed_blocks_v6' not in body:
            raise RuntimeError('GitHub contains stale source imports in ' + name +
                               '; upload/commit the GitHub patch first.')
    threading.Thread(target=watchdog,daemon=True).start()
    (OUT/'SOURCE_PROVENANCE.txt').write_text(
        'Git revision: ' + commit + '\n'
        'block_carrier.py is a reconstructed keyed mapper, not the absent historic FCQR-v6.\n'
        'v6 is EXCLUDED. Recompute all v7-v9 metrics and do not equate them with older versions.\n',
        encoding='utf-8')
    if not os.environ.get('QR_LOCAL_REPO'):
        subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(ROOT),'pytest'],
                       check=True,timeout=600)
    import yaml, pandas as pd
    cdir=ROOT/'configs/methods'
    if not (cdir/'proposed_integer_conv_quad_v9_experimental.yaml').exists():
        raise FileNotFoundError('Quad-v9 config not in GitHub checkout')
    if not (ROOT/'src/qrwatermark/proposed/integer_convolution_quad_v9.py').exists():
        raise FileNotFoundError('Quad-v9 source not in GitHub checkout')
    hosts=sorted((ROOT/'data/hosts/classical').glob('*.bmp'))
    wms=sorted((ROOT/'data/watermarks').glob('*.png'))
    if not hosts or not wms:raise FileNotFoundError('No hosts/watermarks in cloned repo')
    print('GITHUB REVISION',commit,'HOSTS',len(hosts),'WATERMARKS',len(wms),flush=True)
    if not cmd('00_UNIT_TESTS',[sys.executable,'-m','pytest','-q','tests'],limit=900):
        raise RuntimeError('Unit tests failed, stopped before benchmarks')
    methods={
      'quad_v9':'proposed_integer_conv_quad_v9_experimental.yaml',
      'pair_v8':'proposed_integer_conv_pair_v8_experimental.yaml',
      'conv8_v7':'proposed_integer_conv8_v7_experimental.yaml',
      'fcqr_v6':'proposed_fcqr_v6_experimental.yaml',
      'convqr_v5':'proposed_convqr_v5_experimental.yaml'}
    available=[(k,v) for k,v in methods.items()
               if (cdir/v).exists() and (k!='fcqr_v6' or (ROOT/'src/qrwatermark/proposed/fused_convqr_v6.py').exists())]
    family=[{'name':'proposed','config':str((cdir/v).resolve())} for k,v in available]
    v9=family[0]
    papers=['su2014_qr','su2016_hessenberg','su2020_schur','chen2021_qqrd','nha2022_improved_qr']
    papers=[k for k in papers if (cdir/(k+'.yaml')).exists()]
    if len(papers)!=5:raise RuntimeError('Five paper baselines not found in checkout')
    paper=[{'name':k,'config':str((cdir/(k+'.yaml')).resolve())} for k in papers]
    attack_cfg=CFGS/'attack_25.yaml'
    write_yaml(attack_cfg,{'attacks': ([{'name':'clean'}] if SMOKE_ONLY else attacks25())},yaml)
    all_hosts=hosts[:1] if SMOKE_ONLY else hosts
    all_wms=wms[:1] if SMOKE_ONLY else wms

    def conf(h,w,ms,matched,label):
        linkdir=CFGS/'links'/label/(h.stem+'__'+w.stem)
        hs=linkdir/'hosts'; ws=linkdir/'watermarks'
        hs.mkdir(parents=True,exist_ok=True);ws.mkdir(parents=True,exist_ok=True)
        for source,target in ((h,hs/h.name),(w,ws/w.name)):
            if not target.exists():target.symlink_to(source.resolve())
        p=CFGS/(label+'__'+h.stem+'__'+w.stem+'.yaml')
        write_yaml(p,{'match_psnr_to_proposed':matched,'watermark_size':64,
                      'methods':ms,'hosts':str(hs.resolve()),'watermarks':str(ws.resolve()),
                      'attacks':str(attack_cfg.resolve())},yaml)
        return p

    # A clean extraction preflight must evaluate CLEAN ONLY.  Never require
    # BER=0 after JPEG, noise, blur, cropping, or geometric attacks.
    clean_attack_cfg=CFGS/'attack_clean_preflight.yaml'
    write_yaml(clean_attack_cfg, {'attacks':[{'name':'clean'}]}, yaml)
    test_cfg=conf(hosts[0],wms[0],[v9],False,'PREFLIGHT_CLEAN_V2')
    preflight_document=yaml.safe_load(test_cfg.read_text(encoding='utf-8'))
    preflight_document['attacks']=str(clean_attack_cfg.resolve())
    write_yaml(test_cfg,preflight_document,yaml)
    smoke_result=OUT/'00_PREFLIGHT_CLEAN_V2'/'metrics.csv'
    if not cmd('00_PREFLIGHT_CLEAN_V2',[sys.executable,'scripts/run_main_comparison.py',
                '--config',str(test_cfg),'--key',KEY,
                '--run-dir',str(smoke_result.parent)],limit=120,expected=smoke_result):
        raise RuntimeError('Clean preflight execution failed; inspect its job log')
    smoke=pd.read_csv(smoke_result)
    if smoke.empty or not smoke['attack'].eq('clean').all():
        raise RuntimeError('Preflight must contain clean rows only')
    failed_rows=smoke.loc[~(smoke['ber'].eq(0) & smoke['nc'].ge(1 - 1e-12))]
    if not failed_rows.empty:
        failed_rows.to_csv(OUT/'CLEAN_PREFLIGHT_FAILURES.csv',index=False)
        raise RuntimeError('Actual clean decoding is not exact; see CLEAN_PREFLIGHT_FAILURES.csv')
    print('CLEAN PREFLIGHT PASSED: BER=0, NC=1',flush=True)

    # PSNR matching is performed separately for every baseline and image-pair.
    # Each completed metrics.csv survives any later timeout/failure.
    for h in all_hosts:
        for w in all_wms:
            for b in (paper[:1] if SMOKE_ONLY else paper):
                label='01_PAPER_BASELINES_'+b['name']+'__'+h.stem+'__'+w.stem
                run_dir=OUT/'01_PAPER_BASELINES'/b['name']/(h.stem+'__'+w.stem)
                cfg=conf(h,w,[v9,b],not SMOKE_ONLY,label)
                cmd(label,[sys.executable,'scripts/run_main_comparison.py','--config',str(cfg),
                     '--key',KEY,'--run-dir',str(run_dir)],
                     limit=2100,expected=run_dir/'metrics.csv')
                if remaining()<=RESERVE_SECONDS:break
            if remaining()<=RESERVE_SECONDS:break
        if remaining()<=RESERVE_SECONDS:break

    for h in all_hosts:
        for w in all_wms:
            label='02_PROPOSAL_FAMILY__'+h.stem+'__'+w.stem
            rd=OUT/'02_PROPOSAL_FAMILY'/(h.stem+'__'+w.stem)
            c=conf(h,w,family,False,label)
            cmd(label,[sys.executable,'scripts/run_main_comparison.py','--config',str(c),
                  '--key',KEY,'--run-dir',str(rd)],limit=1500,expected=rd/'metrics.csv')
            if remaining()<=RESERVE_SECONDS:break
        if remaining()<=RESERVE_SECONDS:break

    # Genuine within-v9 controls separated from cross-design proposal controls.
    variants={'quad_v9_full':{},'quad_no_scrambling':{'arnold_iterations':0},
              'quad_blue_channel':{'channel':0},'quad_red_channel':{'channel':2}}
    for period in (60,64,68,72,76):
        variants[f'quad_T{period}']={'integer_conv4_period':period}
    for k, filename in available[1:]:
        contents=yaml.safe_load((cdir/filename).read_text())
        variants['cross_design_'+k]=contents['parameters']
    # Smoke ablation subset must also use one image + one watermark.
    if SMOKE_ONLY:variants={'quad_v9_full':{}}
    abl=CFGS/'ablation_v9.yaml'
    smoke_links=CFGS/'links'/'ABLATION_SMOKE'
    if SMOKE_ONLY:
        ah=smoke_links/'hosts'; aw=smoke_links/'watermarks';ah.mkdir(parents=True,exist_ok=True);aw.mkdir(parents=True,exist_ok=True)
        for x,y in ((hosts[0],ah/hosts[0].name),(wms[0],aw/wms[0].name)):
            if not y.exists():y.symlink_to(x.resolve())
    else:
        ah=ROOT/'data/hosts/classical';aw=ROOT/'data/watermarks'
    write_yaml(abl,{'base_config':str((cdir/methods['quad_v9']).resolve()),
                    'hosts':str(ah.resolve()),'watermarks':str(aw.resolve()),
                    'attacks':str(attack_cfg.resolve()),'watermark_size':64,
                    'variants':variants},yaml)
    for variant in variants:
        label='03_ABLATION_'+variant
        rd=OUT/'03_ABLATIONS'/variant
        cmd(label,[sys.executable,'scripts/run_ablation.py','--config',str(abl),'--key',KEY,
                   '--run-dir',str(rd),'--variant',variant],limit=1700,expected=rd/'ablation.csv')
        if remaining()<=RESERVE_SECONDS:break

    rt=CFGS/'runtime.yaml'
    write_yaml(rt,{'methods':[v9,*family[1:],*paper],'repeats':(1 if SMOKE_ONLY else 3)},yaml)
    rd=OUT/'04_RUNTIME'
    cmd('04_RUNTIME',[sys.executable,'scripts/run_runtime.py','--config',str(rt),
                    '--key',KEY,'--run-dir',str(rd)],limit=1500,expected=rd/'runtime.csv')

    fair=join_csv('01_PAPER_BASELINES/*/*/metrics.csv','PAPER_BASELINES_ALL_ROWS.csv',pd)
    family_results=join_csv('02_PROPOSAL_FAMILY/*/metrics.csv','PROPOSAL_FAMILY_ALL_ROWS.csv',pd)
    abl_results=join_csv('03_ABLATIONS/*/ablation.csv','ABLATION_ALL_ROWS.csv',pd)
    rt_results=join_csv('04_RUNTIME/runtime.csv','RUNTIME_ALL_ROWS.csv',pd)
    summary(fair,'PAPER_BASELINES');summary(family_results,'PROPOSAL_FAMILY')
    if not abl_results.empty:
        abl_results.groupby('variant')[['nc','ber','embedding_psnr']].mean().to_csv(OUT/'ABLATION_MEANS.csv')
        abl_results.groupby(['variant','attack'])[['nc','ber']].mean().to_csv(OUT/'ABLATION_BY_ATTACK.csv')
    if not rt_results.empty:
        ms=rt_results.groupby('method')[['embed_seconds','extract_seconds']].median()*1000
        ms['fps_embed_plus_extract']=1000/(ms['embed_seconds']+ms['extract_seconds'])
        ms.to_csv(OUT/'RUNTIME_MEDIAN.csv')
    if not fair.empty and 'matched_psnr_error_db' in fair:
        fair.groupby('method')['matched_psnr_error_db'].agg(['mean','min','max','count']).to_csv(OUT/'PSNR_MATCHING_DIAGNOSTICS.csv')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for name,df in [('paper',fair),('family',family_results)]:
            if df.empty:continue
            values=df[df.attack!='clean'].groupby('method')['nc'].mean().sort_values()
            if values.empty:continue
            fig,ax=plt.subplots(figsize=(10,max(4,len(values)*0.6)))
            values.plot(kind='barh',ax=ax)
            ax.set_xlabel('Mean attacked NC');ax.set_xlim(0,1)
            fig.tight_layout();fig.savefig(OUT/(name+'_NC.pdf'))
            plt.close(fig)
    except Exception as exc:failed.append({'plots':repr(exc)})

except BaseException as exc:
    fatal_error=exc
    failed.append({'pipeline_error':repr(exc),'traceback':traceback.format_exc()})
    print('PIPELINE STOPPED',repr(exc),flush=True)
    print(traceback.format_exc(),flush=True)
finally:
    stop.set()
    if not STALE_RUN:
        save_state();checkpoint()
    print('FINISHED: completed',len(completed),'failed',len(failed),flush=True)
    print('Results ZIP:',ARCHIVE,flush=True)

if fatal_error is not None:
    raise SystemExit(1)
