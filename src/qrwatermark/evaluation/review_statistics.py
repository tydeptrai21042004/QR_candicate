"""Host-clustered bootstrap, paired comparisons and score-based ROC utilities.

A host is the independent experimental unit; attacks, watermarks and random
seeds on that same image are correlated and MUST NOT be treated as independent
bootstrap replicates. No external training of hyperparameters is performed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def host_cluster_ci(rows, metric='ber', confidence=0.95, n_boot=1000, seed=3101):
    """CI on macro-host mean; each host contributes exactly once."""
    df=pd.DataFrame(rows)
    means=df.groupby('host')[metric].mean().astype(float)
    if means.empty:
        raise ValueError('No samples')
    vals=means.to_numpy()
    rng=np.random.default_rng(seed)
    draws=np.mean(rng.choice(vals, size=(n_boot,len(vals)),replace=True),axis=1)
    lo,hi=np.quantile(draws,[(1-confidence)/2,1-(1-confidence)/2])
    return {'mean':float(vals.mean()),'std_host':float(vals.std(ddof=1)) if len(vals)>1 else float('nan'),
            'ci_lower':float(lo),'ci_upper':float(hi),'n_independent_hosts':int(len(vals)),
            'ci_method':'host_cluster_bootstrap','bootstrap_repeats':int(n_boot)}


def paired_host_ci(df, method, reference='green_quad_min_energy',metric='ber',seed=3101,n_boot=1000):
    keys=['host','watermark','attack','attack_params','seed']
    subset=df[df.method.isin([method,reference])].copy()
    a=subset[subset.method==method][keys+[metric]].rename(columns={metric:'candidate'})
    b=subset[subset.method==reference][keys+[metric]].rename(columns={metric:'reference'})
    both=a.merge(b,on=keys,validate='one_to_one')
    if len(both)!=len(a) or len(both)!=len(b):
        raise ValueError(f'{method}: unmatched sample identifiers; cannot claim paired comparison')
    both['difference']=both['candidate']-both['reference']
    return {'candidate':method,'reference':reference,'metric':metric,
            'sign':'candidate_minus_reference', 'n_paired_conditions':len(both),
            **host_cluster_ci(both,metric='difference',n_boot=n_boot,seed=seed)}


def statistical_report(df, out, seed=3101, n_boot=1000):
    from pathlib import Path
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(df)
    by_attack=[]
    for (method,attack,params), sub in df.groupby(['method','attack','attack_params'],dropna=False):
        row={'method':method,'attack':attack,'attack_params':params,'n_conditions':len(sub)}
        for metric in ('ber','nc','embedding_psnr','embedding_ssim'):
            if metric in sub:
                stats=host_cluster_ci(sub,metric,n_boot=n_boot,seed=seed)
                row.update({f'{metric}_{k}':v for k,v in stats.items()})
        by_attack.append(row)
    pd.DataFrame(by_attack).to_csv(out/'per_attack_ci.csv',index=False)
    paired=[]
    if 'green_quad_min_energy' in df.method.unique():
        for method in sorted(set(df.method)-{'green_quad_min_energy'}):
            for metric in ('ber','nc'):
                try:
                    paired.append(paired_host_ci(df,method,metric=metric,seed=seed,n_boot=n_boot))
                except ValueError as e:
                    paired.append({'candidate':method,'reference':'green_quad_min_energy',
                                   'metric':metric,'error':str(e)})
    pd.DataFrame(paired).to_csv(out/'paired_host_ci.csv',index=False)
    return by_attack,paired


def roc_points(positive_scores,negative_scores):
    pos=np.asarray(positive_scores,dtype=float);neg=np.asarray(negative_scores,dtype=float)
    if not len(pos) or not len(neg):
        raise ValueError('ROC requires positive and negative samples')
    # threshold +inf enforces origin, -inf end point; score >= threshold.
    thresholds=np.r_[np.inf,np.sort(np.unique(np.r_[pos,neg]))[::-1],-np.inf]
    fpr=np.array([(neg>=t).mean() for t in thresholds])
    tpr=np.array([(pos>=t).mean() for t in thresholds])
    auc=float(np.trapezoid(tpr,fpr)) if hasattr(np,'trapezoid') else float(np.trapz(tpr,fpr))
    return pd.DataFrame({'threshold':thresholds,'fpr':fpr,'tpr':tpr}),auc


def calibrate_threshold(negative_scores, target_fpr=0.01):
    """Calibrate only on non-test negatives; with '>' threshold ties are rejected."""
    neg=np.asarray(negative_scores,dtype=float)
    if not 0<target_fpr<1 or not len(neg):
        raise ValueError('No calibration negatives / invalid target FPR')
    # Use a conservative empirical threshold: exceed the (1-alpha) quantile.
    threshold=float(np.quantile(neg,1-target_fpr,method='higher'))
    return float(np.nextafter(threshold,np.inf))
