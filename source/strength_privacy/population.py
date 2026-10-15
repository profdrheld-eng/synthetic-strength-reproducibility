"""Fixed artificial study kernels; see docs/KERNEL_PROVENANCE.json."""
import copy
import numpy as np
from . import attacks as a, relationships as t, norms as s, scores as q
MODELS = ("original", "heavy_tails", "shifted_mixed")
METHODS = ("depth8_outcomes_first", "depth8_interleaved")

NOISE = {'exact': (0, 0, 0, 0), 'mild': (1, 1, 1, np.log(1.05)), 'strong': (3, 3, 3, np.log(1.15))}

def population(seed, model):
    if model == 'original':
        return t.make(seed, 'nonlinear')
    data = s.make_artificial(seed)
    latent = data['z'].copy()
    if model == 'heavy_tails':
        rng = np.random.default_rng(seed + 1000000000)
        latent = 0.8 * np.sqrt(3 / 5) * rng.standard_t(5, 393)[:, None] + 0.35 * np.sqrt(3 / 5) * rng.standard_t(5, (393, 3))
        score = latent + t.signal(data, 'nonlinear')
    elif model == 'shifted_mixed':
        data['age'] = 18 + 42 * np.sqrt((data['age'] - 18) / 42)
        data['height'] = np.clip(data['height'] + 3, 145, 205)
        data['mass'] = np.clip(data['mass'] + 8, 45, 125)
        score = latent + 0.75 * t.signal(data, 'nonlinear') + 0.25 * t.signal(data, 'linear')
    else:
        raise ValueError('Unknown fixed population')
    q.encode_scores(data, score)
    return data

def observed(data, level, seed):
    if level not in NOISE:
        raise ValueError('Unknown knowledge error')
    age, height, mass, force = NOISE[level]
    return a.matrix(data) + np.random.default_rng(seed).normal(size=(393, 6)) * np.array([age, height, mass, force, force, force])

def auxiliary_confirmation(release, truth, observation, width):
    queries, counts = np.unique(np.floor(release / width).astype(np.int64), axis=0, return_counts=True)
    queries = queries[counts == 1]
    if not len(queries):
        return None
    true_hits = np.all(queries[:, None, :] == np.floor(truth / width).astype(np.int64)[None, :, :], axis=2)
    aux_hits = np.all(queries[:, None, :] == np.floor(observation / width).astype(np.int64)[None, :, :], axis=2)
    return float(np.mean((true_hits.sum(1) == 1) & (aux_hits.sum(1) == 1) & ((true_hits & aux_hits).sum(1) == 1)))

def evaluate(members, controls, release, dev, method, level, seed, baseline):
    groups = copy.deepcopy(baseline['by_sex'])
    params = dev['parameters'][method]
    obs = [observed(members, level, seed), observed(controls, level, seed + 1000000)]
    for sex in ('female', 'male'):
        mask = [np.array(data['sex']) == sex for data in (members, controls, release)]
        truth = [a.matrix(data)[m] for data, m in zip((members, controls, release), mask)]
        syn = truth[2]
        scale = np.array(dev['scales'][sex])
        group = groups[sex]
        for label, x, o in zip(('member', 'control'), truth[:2], [obs[i][mask[i]] for i in range(2)]):
            group['singling_aux_verified_' + label] = auxiliary_confirmation(syn, x, o, a.WIDTHS[params['singling_width']])
            if level == 'exact':
                continue
            group['link_' + label] = a.bridge(o[:, :3], o[:, 3:], syn, scale[:3], scale[3:], params['link_k'], seed)
            for enhanced in (False, True):
                profile = 'extended' if enhanced else 'demographics'
                for secret, lift in enumerate(('SQ', 'BP', 'DL'), start=3):
                    prediction = a.predict(o, syn, secret, params['inference_k'], scale, enhanced)
                    prefix = 'inference_' + profile + '_' + lift
                    group[prefix + '_' + label + '_mae_kg'] = float(np.mean(np.abs(np.exp(prediction) - np.exp(x[:, secret]))))
                    error = np.abs(np.expm1(prediction - x[:, secret]))
                    for tol in (0.05, 0.1):
                        group[prefix + '_' + str(int(tol * 100)) + '_' + label] = float(np.mean(error <= tol))
    summary = {}
    counts = {sex: members['sex'].count(sex) for sex in groups}
    total = sum(counts.values())
    for key in groups['female']:
        if key.startswith('singling_'):
            continue
        summary[key] = sum((groups[sex][key] * counts[sex] / total for sex in groups))
    queries = sum((groups[sex]['singling_queries'] for sex in groups))
    summary['singling_queries'] = queries
    for key in ('singling_member', 'singling_control', 'singling_aux_verified_member', 'singling_aux_verified_control'):
        summary[key] = sum(((groups[sex][key] or 0) * groups[sex]['singling_queries'] for sex in groups)) / queries if queries else None
    for key in list(summary):
        if key.endswith('_member') and (not ('_prior_' in key or '_random_' in key)):
            other = key[:-7] + '_control'
            if summary[key] is not None and summary[other] is not None:
                summary[key[:-7] + '_excess'] = summary[key] - summary[other]
    return {'summary': summary, 'by_sex': groups}
