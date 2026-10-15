"""Join fixed privacy attacks and utility checks on the same invented release."""
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import sklearn

from strength_privacy import design, norms, population, scores, synthesis
from strength_privacy.validation import validate_outcomes
from .utility import safe_assess

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('cart', 'copy_positive', 'independent_negative')


def fingerprint(data):
    h = hashlib.sha256()
    for key in ('ids', 'sex', 'age', 'height', 'mass', 'strength', 'z'):
        h.update(key.encode())
        value = data[key]
        if isinstance(value, list):
            h.update(json.dumps(value, separators=(',', ':')).encode())
        else:
            x = np.asarray(value, dtype='<f8')
            h.update(str(x.shape).encode())
            h.update(x.tobytes())
    return h.hexdigest()


def inputs():
    provenance = json.loads((ROOT / 'docs/INPUT_PROVENANCE.json').read_text())
    identity = {}
    for name, record in provenance['files'].items():
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if digest != record['sha256']:
            raise ValueError('Copied source or fixture changed: ' + name)
        identity[name] = digest
    with (ROOT / 'fixtures/S4_unchanged_preservation_limits.csv').open() as f:
        expected_limits = {r['metric_key']: float(r['unchanged_limit']) for r in csv.DictReader(f)}
    if scores.LIMITS != expected_limits:
        raise ValueError('Existing preservation limits changed')
    with (ROOT / 'fixtures/S1_reference_constants.csv').open() as f:
        for row in csv.DictReader(f):
            sex, lift = row['stratum'], row['lift']
            expected = tuple(float(row[k]) for k in ('intercept', 'height_cm', 'age_years',
                             'age_squared', 'log_mass_kg', 'log_mass_squared'))
            if norms.COEFFICIENTS[sex][lift] != expected or norms.RSD[sex][lift] != float(row['residual_scale']):
                raise ValueError('Corrected reference changed')
    original = json.loads((ROOT / 'fixtures/privacy_shadow_parameters.json').read_text())
    trained = json.loads((ROOT / 'fixtures/privacy_adaptive_shadow_parameters.json').read_text())
    if original['data_status'] != 'ARTIFICIAL_ONLY':
        raise ValueError('Artificial shadow parameters required')
    return original, trained, identity


def config(models, cohorts, releases, levels):
    if type(cohorts) is not int or not 1 <= cohorts <= 40:
        raise ValueError('Between one and forty independent artificial cohorts required')
    if type(releases) is not int or not 1 <= releases <= 100:
        raise ValueError('Between one and one hundred releases required')
    if (not models or len(set(models)) != len(models)
            or any(m not in population.MODELS for m in models)):
        raise ValueError('Unique known artificial population models required')
    if (not levels or len(set(levels)) != len(levels)
            or any(n not in population.NOISE for n in levels)):
        raise ValueError('Unique known knowledge levels required')


def privacy_evaluation(members, controls, release, original, trained, method, level, seed):
    try:
        values = design.evaluate(members, controls, release, original, trained, method, level, seed)
        for mode in design.MODES:
            validate_outcomes(values[mode])
        return {'status': 'EVALUATED_NOT_A_PRIVACY_PASS', **values}
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {'status': 'NOT_EVALUABLE', 'error_type': type(exc).__name__, 'reason': str(exc)}


def run(*, models=population.MODELS, cohorts=1, releases=1, levels=tuple(population.NOISE)):
    config(models, cohorts, releases, levels)
    original, trained, identity = inputs()
    joint_sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((ROOT / 'joint_strength').glob('*.py'))}
    records = []
    for model in models:
        for cohort in range(cohorts):
            members = population.population(91000000 + cohort, model)
            controls = population.population(92000000 + cohort, model)
            audit = population.population(93000000 + cohort, model)
            negative = population.population(95000000 + cohort, model)
            if any(set(a['ids']) & set(b['ids']) for i, a in enumerate((members, controls, audit, negative))
                   for b in (members, controls, audit, negative)[i+1:]):
                raise ValueError('Artificial source roles overlap')
            original_hashes = [fingerprint(x) for x in (members, controls, audit, negative)]
            for method in population.METHODS:
                for replica in range(releases + 2):
                    kind = KINDS[0 if replica < releases else 1 if replica == releases else 2]
                    release = (synthesis.generate(members, method, 94000000 + 1000 * cohort + replica)
                               if kind == 'cart' else members if kind == 'copy_positive' else negative)
                    release_hash = fingerprint(release)
                    utility = safe_assess(members, release, audit['z'])
                    privacy = {}
                    for level in levels:
                        privacy[level] = privacy_evaluation(members, controls, release, original,
                            trained[model][level], method, level, 96000000 + 1000 * cohort + replica)
                    if (fingerprint(release) != release_hash
                            or [fingerprint(x) for x in (members, controls, audit, negative)] != original_hashes):
                        raise RuntimeError('Evaluation mutated an artificial input')
                    records.append({'model': model, 'cohort': cohort, 'method': method,
                                    'replica': replica, 'kind': kind, 'release_sha256': release_hash,
                                    'reference_seed': 91000000 + cohort, 'control_seed': 92000000 + cohort,
                                    'audit_seed': 93000000 + cohort, 'negative_seed': 95000000 + cohort,
                                    'utility': utility, 'privacy': privacy})
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
           for name, digest in {**identity, **joint_sources}.items()):
        raise RuntimeError('Calculation sources changed during execution')
    report = {'schema_version': 1, 'data_status': 'ARTIFICIAL_ONLY',
              'norm_status': norms.NORM_STATUS, 'limit_status': 'EXISTING_CANDIDATE_LIMITS_UNCHANGED',
              'limits': scores.LIMITS.copy(), 'n_female': 197, 'n_male': 196,
              'models': list(models), 'cohorts_per_model': cohorts, 'releases_per_method': releases,
              'knowledge_levels': list(levels), 'negative_releases_per_cohort': 1,
              'privacy_acceptance': 'NOT_DEFINED', 'joint_pass_rate': None,
              'attack_parameters': 'REUSED_FIXED_CORRECTED_ARTIFICIAL_SHADOW_SELECTION',
              'source_sha256': {**identity, **joint_sources},
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                          'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__},
              'records': records,
              'limits_of_run': ['Complete artificial records only; no empirical data loader.',
                  'Fixed corrected shadow parameters reused, not newly developed.',
                  'One copy and one independent negative release per cohort: integration controls, not the full joint protocol.',
                  'No membership inference, empirical split adapter, missing-data synthesis or nonlinear curve gate implemented.',
                  'Point estimates only; no population confidence intervals or anonymity certification.',
                  'Shared corresponding seeds pair population models; models are not independent repetitions.']}
    validate_report(report)
    report['cohort_summaries'] = summarise(report)
    return report


def validate_report(report):
    models, cohorts, releases, levels = (report[k] for k in
        ('models', 'cohorts_per_model', 'releases_per_method', 'knowledge_levels'))
    config(models, cohorts, releases, levels)
    if (report['data_status'] != 'ARTIFICIAL_ONLY' or report['limits'] != scores.LIMITS
            or report['norm_status'] != norms.NORM_STATUS
            or (report['n_female'], report['n_male']) != (197, 196)):
        raise ValueError('Artificial identity or limits changed')
    if report['privacy_acceptance'] != 'NOT_DEFINED' or report['joint_pass_rate'] is not None:
        raise ValueError('No privacy acceptance threshold has been defined')
    expected = {(m, c, v, r) for m in models for c in range(cohorts)
                for v in population.METHODS for r in range(releases + 2)}
    observed = [(r['model'], r['cohort'], r['method'], r['replica']) for r in report['records']]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError('Incomplete, duplicate or extraneous joint records')
    for row in report['records']:
        r = row['replica']
        if type(r) is not int or type(row['cohort']) is not int:
            raise ValueError('Integer cohort/release indices required')
        for field, base in (('reference_seed', 91000000), ('control_seed', 92000000),
                            ('audit_seed', 93000000), ('negative_seed', 95000000)):
            if type(row[field]) is not int or row[field] != base + row['cohort']:
                raise ValueError('Unexpected source seed')
        if (not isinstance(row['release_sha256'], str) or len(row['release_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['release_sha256'])):
            raise ValueError('Release identity missing')
        if row['kind'] != KINDS[0 if r < releases else 1 if r == releases else 2]:
            raise ValueError('Incorrect release role')
        if set(row['privacy']) != set(levels):
            raise ValueError('Missing knowledge condition')
        for values in row['privacy'].values():
            if values['status'] == 'EVALUATED_NOT_A_PRIVACY_PASS':
                for mode in design.MODES:
                    validate_outcomes(values[mode])
            elif values['status'] != 'NOT_EVALUABLE':
                raise ValueError('Unknown privacy evaluation state')
        utility = row['utility']
        if utility['status'] == 'ARTIFICIAL_CANDIDATE_SCREEN':
            if set(utility['checks']) != {'all', 'female', 'male', 'common'}:
                raise ValueError('Incomplete utility groups')
            common_keys = {'pc1_rmse', 'pc1_bias', 'pc1_correlation_loss',
                           'variance_fraction_change', 'weight_max'}
            for group in utility['checks']:
                keys = common_keys if group == 'common' else set(scores.LIMITS) - common_keys
                values = utility['common_audit'] if group == 'common' else utility['groups'][group]
                if set(utility['checks'][group]) != keys or set(values) != keys:
                    raise ValueError('Missing or extraneous utility criteria')
                for key, value in values.items():
                    flag = utility['checks'][group][key]
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not np.isfinite(value) or type(flag) is not bool
                            or flag != (value <= scores.LIMITS[key] + 1e-12)):
                        raise ValueError('Invalid utility metric or threshold flag')
            if utility['all_passed'] != all(all(v.values()) for v in utility['checks'].values()):
                raise ValueError('Utility pass disagrees with individual criteria')
        elif utility['status'] != 'FAILED_EVALUATION' or utility['all_passed']:
            raise ValueError('Invalid utility failure state')
    json.dumps(report, allow_nan=False)


def summarise(report):
    rows = []
    for model in report['models']:
        for cohort in range(report['cohorts_per_model']):
            for method in population.METHODS:
                for kind in KINDS:
                    group = [r for r in report['records'] if
                             (r['model'], r['cohort'], r['method'], r['kind']) == (model, cohort, method, kind)]
                    privacy = {}
                    for level in report['knowledge_levels']:
                        valid = all(r['privacy'][level]['status'] == 'EVALUATED_NOT_A_PRIVACY_PASS' for r in group)
                        privacy[level] = {'evaluated': sum(r['privacy'][level]['status'] ==
                            'EVALUATED_NOT_A_PRIVACY_PASS' for r in group), 'expected': len(group),
                            'mean_outcomes': {mode: {key: float(np.mean([r['privacy'][level][mode][key]
                                for r in group])) for key in group[0]['privacy'][level][mode]}
                                for mode in design.MODES} if valid else None}
                    rows.append({'model': model, 'cohort': cohort, 'method': method, 'kind': kind,
                        'releases': len(group), 'utility_evaluation_failures': sum(r['utility']['status'] ==
                            'FAILED_EVALUATION' for r in group),
                        'utility_pass_fraction': sum(r['utility']['all_passed'] for r in group) / len(group),
                        'privacy': privacy})
    return rows


def write_report(report, path):
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as handle:
        handle.write(text)
