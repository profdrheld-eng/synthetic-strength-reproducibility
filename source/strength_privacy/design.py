"""Complete fixed designs. Only internally generated artificial cohorts are used."""
import numpy as np
from . import adaptive, population, synthesis, norms
from .statistics import interval
from .validation import validate_outcomes

KINDS = ('cart', 'copy_positive', 'independent_negative')
MODES = ('frozen', 'adapted')


def evaluate(members, controls, release, original, trained, method, level, seed):
    chosen = trained['selected'][method]
    rows = {
        candidate: adaptive.score(members, controls, release, original, method,
                                  level, seed, candidate, trained['scales'])
        for candidate in sorted(set(('frozen', *chosen.values())))
    }
    return {'frozen': rows['frozen'], 'adapted': adaptive.combine(rows, chosen)}


def run_cohort(stage, model, cohort, original, trained):
    if stage not in ('joint', 'earlier') or model not in population.MODELS or cohort not in range(10):
        raise ValueError('Unknown fixed cohort')
    base = 81000000 if stage == 'joint' else 71000000
    replicas, negatives = (100, 10) if stage == 'joint' else (10, 1)
    members = population.population(base + cohort, model)
    controls = population.population(base + 1000000 + cohort, model)
    records = []
    for method in population.METHODS:
        for replica in range(replicas + negatives + 1):
            kind = KINDS[0 if replica < replicas else (1 if replica == replicas else 2)]
            if kind == 'cart':
                release = synthesis.generate(members, method, base + 3000000 + 1000 * cohort + replica)
            elif kind == 'copy_positive':
                release = members
            else:
                negative_seed = (base + 2000000 + 1000 * cohort + replica - replicas - 1
                                 if stage == 'joint' else base + 2000000 + cohort)
                release = population.population(negative_seed, model)
            seed = base + 4000000 + 1000 * cohort + replica
            for level in population.NOISE:
                scores = evaluate(members, controls, release, original,
                                  trained[model][level], method, level, seed)
                records.append({'method': method, 'noise': level, 'kind': kind,
                                'replica': replica, **scores})
    return {'data_status': 'ARTIFICIAL_ONLY', 'norm_status': norms.NORM_STATUS,
            'model': model, 'cohort': cohort, 'reference_seed': base + cohort,
            'control_seed': base + 1000000 + cohort, 'n_female': 197,
            'n_male': 196, 'replicas': replicas, 'negative_releases': negatives,
            'records': records}


def validate_cohorts(cohorts, stage):
    if stage not in ('joint', 'earlier'):
        raise ValueError('Unknown design')
    if (any(type(c['cohort']) is not int for c in cohorts)
            or sorted(c['cohort'] for c in cohorts) != list(range(10))):
        raise ValueError('Exactly ten unique fixed cohorts required')
    replicas, negatives = (100, 10) if stage == 'joint' else (10, 1)
    base = 81000000 if stage == 'joint' else 71000000
    if len({c['model'] for c in cohorts}) != 1 or cohorts[0]['model'] not in population.MODELS:
        raise ValueError('Exactly one known population required')
    for cohort in cohorts:
        if any(type(cohort[key]) is not int for key in
               ('n_female', 'n_male', 'replicas', 'negative_releases', 'reference_seed', 'control_seed')):
            raise ValueError('Integer design metadata required')
        if (cohort['data_status'], cohort['norm_status'], cohort['n_female'], cohort['n_male'],
            cohort['replicas'], cohort['negative_releases']) != (
                'ARTIFICIAL_ONLY', norms.NORM_STATUS, 197, 196, replicas, negatives):
            raise ValueError('Fixed artificial design changed')
        if cohort['reference_seed'] != base + cohort['cohort'] or cohort['control_seed'] != base + 1000000 + cohort['cohort']:
            raise ValueError('Unexpected cohort seeds')
        if len(cohort['records']) != 6 * (replicas + negatives + 1):
            raise ValueError('Incomplete or extraneous records')
        if any(type(row['replica']) is not int for row in cohort['records']):
            raise ValueError('Integer release identifiers required')
        for method in population.METHODS:
            for level in population.NOISE:
                rows = [row for row in cohort['records'] if row['method'] == method and row['noise'] == level]
                if sorted(row['replica'] for row in rows) != list(range(replicas + negatives + 1)):
                    raise ValueError('Missing or duplicate releases')
                for row in rows:
                    expected_kind = KINDS[0 if row['replica'] < replicas else (1 if row['replica'] == replicas else 2)]
                    if row['kind'] != expected_kind:
                        raise ValueError('Release kind disagrees with fixed schedule')
                    for mode in MODES:
                        validate_outcomes(row[mode])


def summarise(cohorts, stage):
    validate_cohorts(cohorts, stage)
    cohorts = sorted(cohorts, key=lambda cohort: cohort['cohort'])
    output = {}
    for level in population.NOISE:
        output[level] = {}
        for method in population.METHODS:
            output[level][method] = {}
            for kind in KINDS:
                rows = [sorted([row for row in c['records'] if row['noise'] == level and
                         row['method'] == method and row['kind'] == kind], key=lambda row: row['replica'])
                        for c in cohorts]
                metrics = {}
                for key in rows[0][0]['frozen']:
                    if key == 'singling_queries':
                        continue
                    means = {mode: [float(np.mean([row[mode][key] for row in group]))
                                    for group in rows] for mode in MODES}
                    metrics[key] = {**{mode: interval(v) for mode, v in means.items()},
                                    'adapted_minus_frozen': interval(np.array(means['adapted']) - means['frozen'])}
                output[level][method][kind] = metrics
    return output
