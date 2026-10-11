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


PAIR_KEYS = ('host', 'watermark', 'attack', 'attack_params', 'seed')


def paired_host_ci(df, method, reference='green_quad_min_energy', metric='ber',
                   seed=3101, n_boot=1000, pairing='strict'):
    """Host-clustered paired CI on exactly common evaluation conditions.

    ``intersection`` is essential for PSNR-eligible comparisons: the proposed
    rows are usually all eligible, while a baseline may be eligible on only a
    subset.  Strict mode remains the default for a complete benchmark.
    Never counts the unmatched rows as zeros or independent replicates.
    """
    if pairing not in ('strict', 'intersection'):
        raise ValueError('pairing must be strict or intersection')
    df = pd.DataFrame(df)
    keys = list(PAIR_KEYS)
    a = df.loc[df.method.eq(method), keys + [metric]].rename(columns={metric:'candidate'})
    b = df.loc[df.method.eq(reference), keys + [metric]].rename(columns={metric:'reference'})
    both = a.merge(b, on=keys, validate='one_to_one')
    if pairing == 'strict' and (len(both) != len(a) or len(both) != len(b)):
        raise ValueError(f'{method}: unmatched sample identifiers; cannot claim strict paired comparison')
    if both.empty:
        raise ValueError(f'{method}: no common evaluation conditions for paired comparison')
    both['difference'] = both['candidate'] - both['reference']
    result = {'candidate':method, 'reference':reference, 'metric':metric,
              'sign':'candidate_minus_reference', 'pairing_policy':pairing,
              'n_paired_conditions':len(both),
              'n_candidate_conditions':len(a), 'n_reference_conditions':len(b),
              'candidate_coverage':float(len(both)/len(a)) if len(a) else 0.0,
              'reference_coverage':float(len(both)/len(b)) if len(b) else 0.0,
              **host_cluster_ci(both,metric='difference',n_boot=n_boot,seed=seed)}
    if result['n_independent_hosts'] < 2:
        result['ci_lower'] = result['ci_upper'] = float('nan')
        result['ci_valid'] = False
        result['caution'] = 'Fewer than two independent matched hosts: no inferential CI'
    else:
        result['ci_valid'] = True
        result['caution'] = 'Host bootstrap; interpret cautiously for small host counts'
    return result


def statistical_report(df, out, seed=3101, n_boot=1000, pair_policy='strict',
                       expected_methods=None):
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
                    paired.append(paired_host_ci(df,method,metric=metric,seed=seed,
                                                 n_boot=n_boot,pairing=pair_policy))
                except ValueError as e:
                    paired.append({'candidate':method,'reference':'green_quad_min_energy',
                                   'metric':metric,'pairing_policy':pair_policy,
                                   'ci_valid':False,'error':str(e)})
    # Explicitly record excluded methods rather than producing an empty CSV
    # that can be mistaken for a successful matched comparison.
    covered={p['candidate'] for p in paired}
    for method in (expected_methods or []):
        if method=='green_quad_min_energy' or method in covered:continue
        for metric in ('ber','nc'):
            paired.append({'candidate':method,'reference':'green_quad_min_energy',
                           'metric':metric,'pairing_policy':pair_policy,
                           'ci_valid':False,
                           'error':'No PSNR-eligible evaluation pairs; no matched CI'})
    pd.DataFrame(paired,columns=['candidate','reference','metric','pairing_policy',
                                  'n_paired_conditions','n_independent_hosts','mean',
                                  'ci_lower','ci_upper','ci_valid','error']).to_csv(
                                      out/'paired_host_ci.csv',index=False)
    return by_attack,paired


def clustered_auc_ci(samples, *, host_col='host', truth_col='truth',
                     score_col='score', n_boot=1000, seed=3101):
    """Macro host-resampling CI for ROC-AUC, preserving all correlated rows.

    Report n_independent_hosts and never interpret wrong keys from one image
    as additional independent hosts.
    """
    df = pd.DataFrame(samples)
    hosts = sorted(df[host_col].unique())
    if len(hosts) < 2:
        return {'n_independent_hosts':len(hosts), 'auc_ci_lower':float('nan'),
                'auc_ci_upper':float('nan'), 'ci_valid':False}
    pieces = {h:df.loc[df[host_col].eq(h)] for h in hosts}
    rng = np.random.default_rng(seed)
    vals=[]
    for _ in range(n_boot):
        draw = rng.choice(hosts, size=len(hosts), replace=True)
        boot = pd.concat([pieces[h] for h in draw], ignore_index=True)
        pos = boot.loc[boot[truth_col].eq(1),score_col].to_numpy()
        neg = boot.loc[boot[truth_col].eq(0),score_col].to_numpy()
        if len(pos) and len(neg):
            vals.append(roc_points(pos,neg)[1])
    if not vals:
        return {'n_independent_hosts':len(hosts), 'auc_ci_lower':float('nan'),
                'auc_ci_upper':float('nan'), 'ci_valid':False}
    lo,hi=np.quantile(vals,[0.025,0.975])
    return {'n_independent_hosts':len(hosts),'auc_ci_lower':float(lo),
            'auc_ci_upper':float(hi),'ci_valid':True,
            'ci_method':'host_cluster_bootstrap','bootstrap_repeats':len(vals)}


def binomial_upper_bound(k, n, confidence=0.95):
    """One-sided Clopper-Pearson upper bound (IID trials only).

    This is a descriptive optimistic bound for correlated key trials; do not
    use it as a host-independent false-positive guarantee.
    """
    from scipy.stats import beta
    if n <= 0 or k < 0 or k > n:
        return float('nan')
    if k == n:
        return 1.0
    return float(beta.ppf(confidence,k+1,n-k))

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
