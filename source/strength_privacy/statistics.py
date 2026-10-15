"""Fixed artificial study kernels; see docs/KERNEL_PROVENANCE.json."""
import numpy as np
from scipy.stats import t, ttest_1samp

def interval(values):
    x = np.asarray(values, dtype=float)
    if x.shape != (10,) or not np.isfinite(x).all():
        raise ValueError('Exactly ten finite cohort means required')
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    half = float(t.ppf(0.975, len(x) - 1) * se)
    indices = np.random.default_rng(31000000).integers(len(x), size=(20000, len(x)))
    boot = x[indices].mean(1)
    return {'n_cohorts': len(x), 'mean': mean, 'cohort_sd': float(x.std(ddof=1)), 'cohort_min': float(x.min()), 'cohort_max': float(x.max()), 't95': [mean - half, mean + half], 'bootstrap95': np.quantile(boot, [0.025, 0.975]).tolist(), 'cohort_means': x.tolist()}

PRIMARY = ('link_excess', *[f'inference_{p}_{lift}_5_excess' for p in ('demographics', 'extended') for lift in ('SQ', 'BP', 'DL')])

def holm(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid p values')
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    result = np.empty(len(p))
    result[order] = adjusted
    return result.tolist()

def pvalue(values):
    x = np.asarray(values, dtype=float)
    if x.shape != (10,) or not np.isfinite(x).all():
        raise ValueError('Ten finite cohort means required')
    if np.all(x == x[0]):
        return 1.0 if x[0] == 0 else 0.0
    return float(ttest_1samp(x, 0).pvalue)

def variance_components(values):
    x = np.asarray(values, dtype=float)
    if x.shape != (10, 10) or not np.isfinite(x).all():
        raise ValueError('Ten cohorts by ten null releases required')
    within = float(x.var(axis=1, ddof=1).mean())
    between = float(x.mean(1).var(ddof=1))
    target = max(0.0, between - within / 10)
    return {'within_release_observation_variance': within, 'cohort_mean_variance': between, 'target_pair_variance': target, 'target_pair_fraction': target / (target + within) if target + within else 0.0, 'within_sd_pp': 100 * np.sqrt(within), 'target_pair_sd_pp': 100 * np.sqrt(target)}
