"""Unchanged 13 utility limits; shared audit profiles, never paired release rows."""
import numpy as np
from scipy.stats import rankdata

from strength_privacy import norms, synthesis
from strength_privacy.scores import LIMITS
from .pc1 import fit_pc1


def validate_data(data):
    n = len(data['ids'])
    if n < 6 or len(set(data['ids'])) != n or len(data['sex']) != n:
        raise ValueError('Unique IDs and consistent nonempty columns required')
    if any(not isinstance(x, str) or not x.startswith('ARTIFICIAL-') for x in data['ids']):
        raise ValueError('Only internally invented IDs supported')
    if set(data['sex']) != {'female', 'male'}:
        raise ValueError('Both known simulation strata required')
    for key in ('age', 'height', 'mass'):
        x = np.asarray(data[key])
        if x.shape != (n,) or not np.isfinite(x).all() or np.any(x <= 0):
            raise ValueError('Finite positive body variables required')
    strength, z = np.asarray(data['strength']), np.asarray(data['z'])
    if strength.shape != (n, 3) or z.shape != (n, 3):
        raise ValueError('Three lift columns required')
    if np.isinf(strength).any() or np.isinf(z).any() or np.any(strength[np.isfinite(strength)] <= 0):
        raise ValueError('Invalid strength or score')
    if not np.array_equal(np.isnan(strength), np.isnan(z)):
        raise ValueError('Missing strength/score pattern disagrees')
    scale = np.array([[norms.RSD[sex][lift] for lift in norms.LIFTS] for sex in data['sex']])
    expected = (np.log(strength) - synthesis.eta_matrix(data)) / scale
    if not np.allclose(expected, z, rtol=0, atol=1e-10, equal_nan=True):
        raise ValueError('Strength and scores disagree with corrected published reference')


def distribution(a, b):
    if len(a) < 3 or len(b) < 3:
        raise ValueError('Insufficient complete profiles')
    sd = a.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError('Degenerate reference distribution')
    return {'mean': float(np.max(abs(b.mean(0) - a.mean(0)))),
            'sd_ratio': float(np.max(abs(b.std(0, ddof=1) / sd - 1))),
            'tail': float(np.max(abs(np.quantile(b, [.05, .95], axis=0)
                                    - np.quantile(a, [.05, .95], axis=0))))}


def correlation(a, b, ranked=False):
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    if ranked:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def relationships(reference, release, model):
    result = {}
    for sex in ('female', 'male'):
        masks = [np.array(d['sex']) == sex for d in (reference, release)]
        score_arrays = [np.column_stack([d['z'], model.transform(d['z'])])
                        for d in (reference, release)]
        for j, outcome in enumerate((*norms.LIFTS, 'PC1')):
            for body in ('age', 'height', 'mass'):
                for label, ranked in (('pearson', False), ('spearman', True)):
                    values = [correlation(x[m, j], np.asarray(d[body])[m], ranked)
                              for d, x, m in zip((reference, release), score_arrays, masks)]
                    result[f'{sex}/{outcome}/{body}/{label}'] = {
                        'reference': values[0], 'release': values[1],
                        'absolute_change': None if None in values else abs(values[1] - values[0])}
    return result


def assess(reference, release, audit_z):
    validate_data(reference)
    validate_data(release)
    a, b = reference['z'], release['z']
    original, new = fit_pc1(a), fit_pc1(b)
    original_score, frozen_release = original.transform(a), original.transform(b)
    groups, checks = {}, {}
    for sex in ('all', 'female', 'male'):
        ma = np.ones(len(a), dtype=bool) if sex == 'all' else np.array(reference['sex']) == sex
        mb = np.ones(len(b), dtype=bool) if sex == 'all' else np.array(release['sex']) == sex
        marginals = [distribution(a[ma & np.isfinite(a[:, j]), j], b[mb & np.isfinite(b[:, j]), j])
                     for j in range(3)]
        ca, cb = ma & np.isfinite(a).all(1), mb & np.isfinite(b).all(1)
        pc = distribution(original_score[ca], frozen_release[cb])
        group = {'z_mean_max': max(v['mean'] for v in marginals),
                 'z_sd_ratio_max': max(v['sd_ratio'] for v in marginals),
                 'z_tail_max': max(v['tail'] for v in marginals),
                 'correlation_max': float(np.max(abs(np.corrcoef(a[ca], rowvar=False)
                                                     - np.corrcoef(b[cb], rowvar=False)))),
                 'frozen_pc1_mean': pc['mean'], 'frozen_pc1_sd_ratio': pc['sd_ratio'],
                 'frozen_pc1_tail': pc['tail'],
                 'missing_fraction_change': abs(float((~cb[mb]).mean() - (~ca[ma]).mean()))}
        if not np.isfinite(list(group.values())).all():
            raise ValueError('Nonfinite utility endpoint')
        groups[sex] = group
        checks[sex] = {key: value <= LIMITS[key] + 1e-12 for key, value in group.items()}
    p, q = original.transform(audit_z), new.transform(audit_z)
    valid = np.isfinite(p) & np.isfinite(q)
    if valid.sum() < 3:
        raise ValueError('Insufficient common audit profiles')
    difference = q[valid] - p[valid]
    common = {'pc1_rmse': float(np.sqrt(np.mean(difference ** 2))),
              'pc1_bias': abs(float(difference.mean())),
              'pc1_correlation_loss': float(1 - np.corrcoef(p[valid], q[valid])[0, 1]),
              'variance_fraction_change': abs(new.variance_fraction - original.variance_fraction),
              'weight_max': float(np.max(abs(np.asarray(new.weights) - original.weights)))}
    if not np.isfinite(list(common.values())).all():
        raise ValueError('Nonfinite common audit endpoint')
    checks['common'] = {key: value <= LIMITS[key] + 1e-12 for key, value in common.items()}
    parameter_records = {role: {'means': list(model.means), 'sds': list(model.sds),
        'weights': list(model.weights), 'raw_mean': model.raw_mean, 'raw_sd': model.raw_sd,
        'n_complete': model.n_complete, 'variance_fraction': model.variance_fraction}
        for role, model in (('reference', original), ('release', new))}
    return {'status': 'ARTIFICIAL_CANDIDATE_SCREEN', 'groups': groups,
            'common_audit': common, 'checks': checks,
            'all_passed': all(all(v.values()) for v in checks.values()),
            'complete_reference': original.n_complete, 'complete_release': new.n_complete,
            'audit_complete': int(valid.sum()), 'audit_excluded': int(len(p) - valid.sum()),
            'reference_pc1_variance_fraction': original.variance_fraction,
            'release_pc1_variance_fraction': new.variance_fraction,
            'pc1_parameters': parameter_records, 'lift_order': list(norms.LIFTS),
            'relationship_diagnostics': relationships(reference, release, original),
            'relationship_gate': 'NOT_APPLIED_NO_NEW_LIMITS'}


def safe_assess(reference, release, audit_z):
    try:
        return assess(reference, release, audit_z)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {'status': 'FAILED_EVALUATION', 'error_type': type(exc).__name__,
                'reason': str(exc), 'all_passed': False}
