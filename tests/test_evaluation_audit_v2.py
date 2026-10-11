"""Coverage, calibration isolation, duplicate detection and honest security CIs."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import pytest

from qrwatermark.evaluation.review_statistics import (
    paired_host_ci,statistical_report,clustered_auc_ci,binomial_upper_bound)
from qrwatermark.evaluation.review_protocol import audit_host_corpus
from qrwatermark.evaluation.fairness import calibrate_global_baseline


def mock_rows():
    rows=[]
    for host in ('A','B','C'):
        for attack in ('clean','jpeg'):
            for method in ('green_quad_min_energy','baseline_1'):
                # Only A and B baseline rows survive PSNR matching. C proposed
                # rows remain eligible, so strict matching must *not* be used.
                if method=='baseline_1' and host=='C':continue
                rows.append(dict(method=method,host=host,watermark='w1',attack=attack,
                    attack_params='{}',seed=None,ber=0.3 if method=='baseline_1' else 0.2,
                    nc=0.7 if method=='baseline_1' else 0.8,
                    embedding_psnr=51.0,embedding_ssim=0.98))
    return pd.DataFrame(rows)


def test_intersection_pairstatistics_preserve_cluster_units(tmp_path):
    df=mock_rows()
    with pytest.raises(ValueError,match='unmatched sample identifiers'):
        paired_host_ci(df,'baseline_1',metric='ber',pairing='strict')
    result=paired_host_ci(df,'baseline_1',metric='ber',pairing='intersection',n_boot=200)
    assert result['n_paired_conditions']==4
    assert result['n_independent_hosts']==2
    assert result['n_candidate_conditions']==4
    assert result['n_reference_conditions']==6
    assert result['ci_valid']
    assert result['mean']==pytest.approx(0.1)
    statistical_report(df,tmp_path,pair_policy='intersection',n_boot=200)
    output=pd.read_csv(tmp_path/'paired_host_ci.csv')
    assert not output.error.notna().any() if 'error' in output else True
    assert set(output.metric)=={'ber','nc'}


def test_intersection_one_host_is_not_claimed_as_inferential_ci():
    df=mock_rows()
    df=df[~((df.method=='baseline_1') & (df.host=='B'))]
    result=paired_host_ci(df,'baseline_1',pairing='intersection')
    assert not result['ci_valid']
    assert np.isnan(result['ci_lower'])


def test_duplicate_decoded_images_are_rejected(tmp_path):
    a=tmp_path/'a.png';b=tmp_path/'b.bmp';c=tmp_path/'c.png'
    img=np.random.default_rng(9).integers(0,255,(32,32,3),dtype=np.uint8)
    cv2.imwrite(str(a),img);cv2.imwrite(str(b),img)
    with pytest.raises(ValueError,match='Duplicate decoded'):
        audit_host_corpus([a,b],min_hosts=2,require_shape=(32,32))
    cv2.imwrite(str(c),255-img)
    checks=audit_host_corpus([a,c],min_hosts=2,require_shape=(32,32))
    assert len(checks)==2


def test_host_count_guard(tmp_path):
    with pytest.raises(ValueError,match='independent REAL'):
        audit_host_corpus([],min_hosts=12)


@dataclass(frozen=True)
class SyntheticConfig:
    quant_step: float


class SyntheticBaseline:
    name='synthetic_strength_method'
    strength_field='quant_step'
    def __init__(self, config):self.config=config
    def embed(self,host,watermark,*,key):
        # Simulated embedding PSNR depends on scale; never reads attacks/test.
        strength=float(self.config.quant_step)
        img=host.astype(np.int16).copy()
        img[::2,::2,:]+=round(strength)
        return SimpleNamespace(image=np.clip(img,0,255).astype(np.uint8))


def test_global_calibration_reproducible_validation_only():
    rng=np.random.default_rng(17)
    host=rng.integers(30,210,(48,48,3),dtype=np.uint8)
    watermark=np.zeros((8,8),dtype=np.uint8)
    from qrwatermark.evaluation.metrics import psnr
    baseline=SyntheticBaseline(SyntheticConfig(quant_step=4.0))
    desired=psnr(host,SyntheticBaseline(SyntheticConfig(quant_step=2.0)).embed(
        host,watermark,key=b'k').image)
    tuned,diag=calibrate_global_baseline(baseline,[(host,watermark,desired)],
                                           key=b'k',psnr_tolerance=0.2)
    assert diag['validation_psnr_matched_pairs']==1
    assert diag['selected_scale']>0
    assert abs(psnr(host,tuned.embed(host,watermark,key=b'k').image)-desired)<0.2


def test_security_host_level_uncertainty_not_wrong_key_count():
    df=pd.DataFrame([{'host':h,'truth':truth,'score':score} for h in ('H1','H2','H3')
        for truth,score in [(1,0.9),(1,0.8),(0,0.2),(0,0.3)]])
    result=clustered_auc_ci(df,n_boot=200)
    assert result['n_independent_hosts']==3
    assert result['ci_valid']
    assert binomial_upper_bound(0,1000)<binomial_upper_bound(0,3)
