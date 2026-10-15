"""Fixed artificial study kernels; see docs/KERNEL_PROVENANCE.json."""
import numpy as np

WIDTHS = (np.array([1, 1, 1, *[np.log(1.05)] * 3]), np.array([5, 5, 5, *[np.log(1.1)] * 3]))

def matrix(data):
    return np.column_stack([data['age'], data['height'], data['mass'], np.log(data['strength'])])

def check_disjoint(a, b):
    if set(a['ids']) & set(b['ids']):
        raise ValueError('Member/control identity overlap')

def neighbours(target, release, scale, k):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('Neighbour count must be a positive integer')
    if target.ndim != 2 or release.ndim != 2 or not len(release) or not len(target):
        raise ValueError('Nonempty two-dimensional arrays required')
    if target.shape[1] != release.shape[1] or np.shape(scale) != (target.shape[1],):
        raise ValueError('Incompatible feature dimensions')
    if not np.isfinite(scale).all():
        raise ValueError('Finite scales required')
    if not np.isfinite(target).all() or not np.isfinite(release).all() or np.any(scale <= 0):
        raise ValueError('Invalid nearest-neighbour input')
    distance = np.sum(((target[:, None, :] - release[None, :, :]) / scale) ** 2, axis=2)
    bound = np.partition(distance, min(k, len(release)) - 1, axis=1)[:, min(k, len(release)) - 1]
    selected = distance <= bound[:, None] + 1e-12
    return selected / selected.sum(1)[:, None]

def bridge(a, b, release, scale_a, scale_b, k, seed):
    order = np.random.default_rng(seed).permutation(len(b))
    truth = np.argsort(order)
    wa = neighbours(a, release[:, :3], scale_a, k)
    wb = neighbours(b[order], release[:, 3:], scale_b, k)
    scores = wa @ wb.T
    maxima = np.isclose(scores, scores.max(1)[:, None], rtol=0, atol=1e-12)
    return float(np.mean(maxima[np.arange(len(a)), truth] / maxima.sum(1)))

def predict(target, release, secret, k, scale, enhanced):
    known = [j for j in range(6 if enhanced else 3) if j != secret]
    weights = neighbours(target[:, known], release[:, known], scale[known], k)
    return weights @ release[:, secret]

def predicates(release, targets, widths):
    queries, counts = np.unique(np.floor(release / widths).astype(np.int64), axis=0, return_counts=True)
    queries = queries[counts == 1]
    if not len(queries):
        return {'queries': 0, 'successes': 0, 'success_rate': None}
    cells = np.floor(targets / widths).astype(np.int64)
    matches = np.all(queries[:, None, :] == cells[None, :, :], axis=2).sum(1)
    successes = int(np.sum(matches == 1))
    return {'queries': len(queries), 'successes': successes, 'success_rate': successes / len(queries)}

def evaluate(members, controls, release, params, scales, priors, seed):
    check_disjoint(members, controls)
    groups = {}
    rng = np.random.default_rng(seed)
    for sex in ('female', 'male'):
        arrays = [matrix(data)[np.array(data['sex']) == sex] for data in (members, controls, release)]
        m, c, syn = arrays
        scale = np.array(scales[sex])
        group = {}
        if len(m) != len(c):
            raise ValueError('Unequal member/control populations')
        singles = [predicates(syn, x, WIDTHS[params['singling_width']]) for x in (m, c)]
        group['singling_queries'] = singles[0]['queries']
        for label, value in zip(('member', 'control'), singles):
            group['singling_' + label] = value['success_rate']
        group['link_member'] = bridge(m[:, :3], m[:, 3:], syn, scale[:3], scale[3:], params['link_k'], seed)
        group['link_control'] = bridge(c[:, :3], c[:, 3:], syn, scale[:3], scale[3:], params['link_k'], seed)
        group['link_chance'] = 1 / len(m)
        for enhanced in (False, True):
            profile = 'extended' if enhanced else 'demographics'
            for secret, lift in enumerate(('SQ', 'BP', 'DL'), start=3):
                prefix = 'inference_' + profile + '_' + lift
                for label, x in (('member', m), ('control', c)):
                    prediction = predict(x, syn, secret, params['inference_k'], scale, enhanced)
                    prior = np.full(len(x), priors[sex][secret - 3])
                    random = rng.choice(syn[:, secret], size=len(x))
                    error = np.abs(np.expm1(prediction - x[:, secret]))
                    group[prefix + '_' + label + '_mae_kg'] = float(np.mean(np.abs(np.exp(prediction) - np.exp(x[:, secret]))))
                    for tol in (0.05, 0.1):
                        key = prefix + '_' + str(int(tol * 100))
                        group[key + '_' + label] = float(np.mean(error <= tol))
                        group[key + '_prior_' + label] = float(np.mean(np.abs(np.expm1(prior - x[:, secret])) <= tol))
                        group[key + '_random_' + label] = float(np.mean(np.abs(np.expm1(random - x[:, secret])) <= tol))
        groups[sex] = group
    summary = {}
    counts = {sex: members['sex'].count(sex) for sex in groups}
    total = sum(counts.values())
    for key in groups['female']:
        if key.startswith('singling_'):
            continue
        summary[key] = sum((groups[sex][key] * counts[sex] / total for sex in groups))
    queries = sum((groups[sex]['singling_queries'] for sex in groups))
    summary['singling_queries'] = queries
    for label in ('member', 'control'):
        summary['singling_' + label] = sum(((groups[sex]['singling_' + label] or 0) * groups[sex]['singling_queries'] for sex in groups)) / queries if queries else None
    for key in list(summary):
        if key.endswith('_member') and (not ('_prior_' in key or '_random_' in key)):
            other = key[:-7] + '_control'
            if summary[key] is not None and summary[other] is not None:
                summary[key[:-7] + '_excess'] = summary[key] - summary[other]
    return {'summary': summary, 'by_sex': groups}
