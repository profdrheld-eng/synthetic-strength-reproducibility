"""Fixed artificial study kernels; see docs/KERNEL_PROVENANCE.json."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from . import attacks as a, population as r, synthesis as g

SHADOW_SEEDS = tuple(range(61000000, 61000003))

TEST_SEEDS = tuple(range(71000000, 71000010))

CANDIDATES = ('frozen', 'nn1', 'nn5', 'nn15', 'trees')

def infer(observation, release, secret, enhanced, scale, kind, seed):
    known = [j for j in range(6 if enhanced else 3) if j != secret]
    if kind != 'trees':
        return a.predict(observation, release, secret, int(kind[2:]), scale, enhanced)
    model = ExtraTreesRegressor(n_estimators=32, max_depth=8, min_samples_leaf=3, random_state=seed, n_jobs=1)
    model.fit(release[:, known], release[:, secret])
    return model.predict(observation[:, known])

def score(members, controls, release, dev, method, level, seed, kind, scales):
    a.check_disjoint(members, controls)
    if kind == 'frozen':
        baseline = a.evaluate(members, controls, release, dev['parameters'][method], dev['scales'], dev['priors'], seed)
        full = r.evaluate(members, controls, release, dev, method, level, seed, baseline)['summary']
        return {k: v for k, v in full.items() if k.startswith('link_') or k.startswith('singling_') or (k.startswith('inference_') and '_prior_' not in k and ('_random_' not in k))}
    observations = [r.observed(members, level, seed), r.observed(controls, level, seed + 1000000)]
    groups = []
    for sex in ('female', 'male'):
        masks = [np.array(d['sex']) == sex for d in (members, controls, release)]
        truth = [a.matrix(d)[mask] for d, mask in zip((members, controls, release), masks)]
        syn = truth[2]
        scale = np.array(scales[sex])
        row = {}
        width = a.WIDTHS[1]
        row['singling_queries'] = a.predicates(syn, truth[0], width)['queries']
        k = 5 if kind == 'trees' else int(kind[2:])
        for label, t, o in zip(('member', 'control'), truth[:2], [observations[i][masks[i]] for i in range(2)]):
            row['singling_' + label] = a.predicates(syn, t, width)['success_rate']
            row['singling_aux_verified_' + label] = r.auxiliary_confirmation(syn, t, o, width)
            row['link_' + label] = a.bridge(o[:, :3], o[:, 3:], syn, scale[:3], scale[3:], k, seed)
            for enhanced in (False, True):
                profile = 'extended' if enhanced else 'demographics'
                for secret, lift in enumerate(('SQ', 'BP', 'DL'), 3):
                    pred = infer(o, syn, secret, enhanced, scale, kind, seed)
                    prefix = 'inference_' + profile + '_' + lift
                    row[prefix + '_' + label + '_mae_kg'] = float(np.mean(abs(np.exp(pred) - np.exp(t[:, secret]))))
                    for tol in (5, 10):
                        row[prefix + '_' + str(tol) + '_' + label] = float(np.mean(abs(np.expm1(pred - t[:, secret])) <= tol / 100))
        row['link_chance'] = 1 / len(truth[0])
        groups.append(row)
    output = {}
    for key in groups[0]:
        weights = [groups[i]['singling_queries'] for i in range(2)] if key.startswith('singling_') else [197, 196]
        if key == 'singling_queries':
            output[key] = sum(weights)
        else:
            output[key] = sum(((groups[i][key] or 0) * weights[i] for i in range(2))) / sum(weights) if sum(weights) else None
    for key in list(output):
        if key.endswith('_member') and output[key] is not None:
            output[key[:-7] + '_excess'] = output[key] - output[key[:-7] + '_control']
    return output

def scales_from(shadows, level):
    result = {}
    for sex in ('female', 'male'):
        rows = np.concatenate([a.matrix(d)[np.array(d['sex']) == sex] for d in shadows])
        noise = r.NOISE[level]
        sd = np.array([*noise[:3], *[noise[3]] * 3])
        result[sex] = np.sqrt(rows.var(0, ddof=1) + sd ** 2).tolist()
    return result

def families(row):
    return ['link', 'singling', 'singling_aux_verified', *['inference_' + p + '_' + lift for p in ('demographics', 'extended') for lift in ('SQ', 'BP', 'DL')]]

def family_key(family):
    return family + ('_5_excess' if family.startswith('inference_') else '_excess')

def development(model, level, dev):
    shadows = [r.population(seed, model) for seed in SHADOW_SEEDS]
    scales = scales_from(shadows, level)
    scores = {k: [] for k in CANDIDATES}
    for c, m in enumerate(shadows):
        control = r.population(62000000 + c, model)
        for method in r.METHODS:
            release = g.generate(m, method, 63000000 + c)
            for kind in CANDIDATES:
                scores[kind].append((method, score(m, control, release, dev, method, level, 64000000 + c, kind, scales)))
    selected = {}
    means = {}
    for method in r.METHODS:
        selected[method] = {}
        means[method] = {}
        for family in families(scores['frozen'][0][1]):
            key = family_key(family)
            values = {kind: float(np.mean([row[key] for meth, row in scores[kind] if meth == method])) for kind in CANDIDATES}
            selected[method][family] = max(CANDIDATES, key=lambda kind: values[kind])
            means[method][family] = values
    return {'model': model, 'noise': level, 'scales': scales, 'selected': selected, 'shadow_excess_means': means, 'shadow_seeds': SHADOW_SEEDS, 'control_seeds': list(range(62000000, 62000003))}

def combine(rows, selected):
    result = {}
    for key in rows['frozen']:
        if key == 'singling_queries':
            family = 'singling'
        else:
            family = next((f for f in sorted(selected, key=len, reverse=True) if key.startswith(f + '_')), None)
        result[key] = rows[selected[family]][key] if family else rows['frozen'][key]
    return result
