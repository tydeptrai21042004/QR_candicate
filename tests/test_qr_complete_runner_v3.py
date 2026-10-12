"""Regression guards for separation, completeness, and resumable archives."""
from pathlib import Path
import importlib.util
import json
import sys
import zipfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_green_quad_ablation import (STUDIES,read_dataset,variant_method,
                                     ABLATION_VARIANTS,HYPERPARAMETER_VARIANTS)
from run_qr_complete_suite import Job,make_jobs,accepted,checkpoint,corpus_fingerprint


def test_studies_are_disjoint_except_identical_final():
    assert set(ABLATION_VARIANTS)&set(HYPERPARAMETER_VARIANTS)=={'FINAL_GREEN_T66'}
    assert all('period' not in name and 'arnold' not in name for name in ABLATION_VARIANTS)
    assert all('channel' not in name and 'greedy' not in name for name in HYPERPARAMETER_VARIANTS)
    assert HYPERPARAMETER_VARIANTS['period_64']=={'integer_conv4_period':64}
    assert HYPERPARAMETER_VARIANTS['period_68']=={'integer_conv4_period':68}


def test_final_unchanged_in_both_studies():
    a=variant_method('FINAL_GREEN_T66','ablation').config
    b=variant_method('FINAL_GREEN_T66','hyperparameters').config
    assert a==b
    assert a.integer_conv4_period==66 and a.channel==1 and a.projection=='optimal'


def test_partitions_are_host_disjoint():
    val,v=read_dataset('quick',partition='validation',min_hosts=6)
    tst,t=read_dataset('quick',partition='test',min_hosts=6)
    assert not set(v['host_groups']['validation']) & set(t['host_groups']['test'])
    assert val[0][0] in {Path(x).stem for x in v['host_groups']['validation']}
    assert tst[0][0] in {Path(x).stem for x in t['host_groups']['test']}
    assert v['exploratory_under_12']


def _args(mode='full'):
    return SimpleNamespace(mode=mode,min_hosts=6,repeats=3,security_wrong_keys=6,
                           neural_checkpoint=None,neural_runner=None,neural_method='example')


def test_full_job_list_has_all_required_outputs(tmp_path):
    jobs=make_jobs(_args('full'),tmp_path,ROOT/'data/hosts/classical')
    names={j.name for j in jobs}
    assert names=={'maintained_tests','ablation','hyperparameter_analysis',
       'reviewer_security','reviewer_attacks','reviewer_optimality',
       'reviewer_capacity','reviewer_runtime',
       'paper_baselines_standard','paper_baselines_256bit'}
    assert all(j.required for j in jobs)
    assert all(j.group!='neural' for j in jobs)
    descriptions={j.name:j for j in jobs}
    for name in ('ablation','hyperparameter_analysis'):
        assert ('--partition','all') in tuple(zip(descriptions[name].cmd,descriptions[name].cmd[1:]))
    for name in ('paper_baselines_standard','paper_baselines_256bit'):
        assert all('matched_only' not in output for output in descriptions[name].required)


def test_matched_psnr_reviewer_job_opt_in_only(tmp_path):
    args=_args('full')
    assert 'reviewer_baselines_matched' not in {j.name for j in make_jobs(args,tmp_path,ROOT/'data/hosts/classical')}
    args.include_matched_baselines=True
    assert 'reviewer_baselines_matched' in {j.name for j in make_jobs(args,tmp_path,ROOT/'data/hosts/classical')}


def test_smoke_explicitly_excludes_long_paper_comparisons(tmp_path):
    names={j.name for j in make_jobs(_args('smoke'),tmp_path,ROOT/'data/hosts/classical')}
    assert 'paper_baselines_standard' not in names
    assert 'ablation' in names and 'hyperparameter_analysis' in names


def test_outputs_must_exist_and_not_be_empty(tmp_path):
    j=Job('x','tests',('python','anything'),('results.csv',))
    assert not accepted(j,tmp_path)
    f=tmp_path/'results.csv';f.touch()
    assert not accepted(j,tmp_path)
    f.write_text('header\n')
    assert accepted(j,tmp_path)


def test_checkpoint_atomic_and_zip_integrity(tmp_path):
    import threading
    out=tmp_path/'run';out.mkdir();(out/'f.csv').write_text('value\n1\n')
    z=tmp_path/'result.zip'
    checkpoint(out,z,threading.Lock())
    assert not z.with_suffix('.zip.tmp').exists()
    with zipfile.ZipFile(z) as arc:
        assert arc.testzip() is None
        assert arc.read('run/f.csv')==b'value\n1\n'


def test_host_fingerprint_deterministic():
    folder=ROOT/'data/hosts/classical'
    x,n=corpus_fingerprint(folder)
    assert (x,n)==corpus_fingerprint(folder)
    assert n==6

def test_runtime_hidden_error_is_not_success(tmp_path):
    p=tmp_path/'reviewer_full/runtime/runtime.csv';p.parent.mkdir(parents=True)
    p.write_text('method,error\ngreen_quad_min_energy,RuntimeError(test)\n')
    job=Job('reviewer_runtime','review',('python','...'),
            ('reviewer_full/runtime/runtime.csv',))
    assert not accepted(job,tmp_path)
