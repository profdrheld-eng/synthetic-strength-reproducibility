"""Fixed outcome schema, checked before any manuscript output is accepted."""
import math
from . import population

EXPECTED_OUTCOMES = frozenset(
    ['link_member', 'link_control', 'link_chance', 'link_excess', 'singling_queries']
    + [family + '_' + suffix for family in ('singling', 'singling_aux_verified')
       for suffix in ('member', 'control', 'excess')]
    + [f'inference_{profile}_{lift}_{role}_mae_kg'
       for profile in ('demographics', 'extended') for lift in ('SQ', 'BP', 'DL')
       for role in ('member', 'control')]
    + [f'inference_{profile}_{lift}_{tolerance}_{role}'
       for profile in ('demographics', 'extended') for lift in ('SQ', 'BP', 'DL')
       for tolerance in (5, 10) for role in ('member', 'control', 'excess')])


def validate_outcomes(metrics):
    if set(metrics) != EXPECTED_OUTCOMES:
        raise ValueError('All 59 fixed outcomes are required; no additional outcomes allowed')
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Finite numeric outcome required: ' + key)
        if key == 'singling_queries':
            if type(value) is not int or not 1 <= value <= 393:
                raise ValueError('Invalid predicate denominator')
        elif key.endswith('mae_kg'):
            if value < 0:
                raise ValueError('Negative absolute error')
        elif key.endswith('_excess'):
            prefix = key[:-7]
            if abs(value - (metrics[prefix + '_member'] - metrics[prefix + '_control'])) > 1e-12:
                raise ValueError('Signed difference does not match absolute rates')
        elif not 0 <= value <= 1:
            raise ValueError('Success fraction outside [0,1]')
    if abs(metrics['link_chance'] - 2 / 393) > 1e-12:
        raise ValueError('Incorrect stratum-weighted chance rate')
    for profile in ('demographics', 'extended'):
        for lift in ('SQ', 'BP', 'DL'):
            for role in ('member', 'control'):
                prefix = f'inference_{profile}_{lift}_'
                if metrics[prefix + '5_' + role] > metrics[prefix + '10_' + role] + 1e-12:
                    raise ValueError('5% success exceeds 10% success')


def validate_all(cohorts, stage, model):
    from . import design
    design.validate_cohorts(cohorts, stage)
    if model not in population.MODELS or any(row['model'] != model for row in cohorts):
        raise ValueError('Population identity mismatch')
