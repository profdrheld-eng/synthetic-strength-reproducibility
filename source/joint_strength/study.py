"""Fixed multicohort artificial study. No person-data input or method search."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta, t, ttest_1samp

from strength_privacy import norms, population, scores, synthesis
from strength_privacy.statistics import PRIMARY, holm
from .runner import ROOT, fingerprint, inputs, privacy_evaluation, write_report
from .utility import safe_assess

COHORTS, RELEASES, NATURAL_REPLICAS, NEGATIVES = 10, 100, 100, 10
BOOTSTRAPS = 10000


def role_seeds(cohort):
    return {'reference': 111000000 + cohort, 'control': 112000000 + cohort,
            'audit': 113000000 + cohort, 'negative_base': 117000000 + 1000 * cohort}


def bootstrap(reference, seed):
    rng = np.random.default_rng(seed)
    sex = np.array(reference['sex'])
    indices = np.concatenate([rng.choice(np.flatnonzero(sex == group), int((sex == group).sum()),
                                         replace=True) for group in ('female', 'male')])
    sample = {key: [value[i] for i in indices] if isinstance(value, list) else value[indices].copy()
              for key, value in reference.items()}
    # These are draw identifiers, not new independent people or privacy controls.
    sample['ids'] = [f'ARTIFICIAL-BOOTSTRAP-{seed}-{i}' for i in range(len(indices))]
    return sample


def estimate(values, probability=False):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Complete finite cohort values required')
    if probability and np.any((x < 0) | (x > 1)):
        raise ValueError('Cohort probabilities outside [0,1]')
    result = {'n_cohorts': len(x), 'mean': float(x.mean()), 'cohort_means': x.tolist(),
              'cohort_sd': float(x.std(ddof=1)) if len(x) > 1 else None,
              't95': None, 'bootstrap95': None}
    if len(x) > 1:
        half = float(t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)))
        result['t95'] = [float(x.mean()-half), float(x.mean()+half)]
        index = np.random.default_rng(131000000).integers(len(x), size=(BOOTSTRAPS, len(x)))
        result['bootstrap95'] = np.quantile(x[index].mean(1), [.025, .975]).tolist()
    if probability:
        successes, n = int(np.count_nonzero(x > 0)), len(x)
        result['cohort_any_positive_exact95'] = [
            0. if successes == 0 else float(beta.ppf(.025, successes, n-successes+1)),
            1. if successes == n else float(beta.ppf(.975, successes+1, n-successes))]
        result['zero_boundary_note'] = (
            'Degenerate bootstrap intervals do not establish zero population probability. '
            'Exact interval is for the different endpoint: a cohort yielding at least one pass, '
            'not the per-release pass probability. t intervals are untruncated approximations.')
    return result


def utility_summary(rows):
    if not rows:
        raise ValueError('No intended utility evaluations')
    good = [r for r in rows if r['status'] == 'ARTIFICIAL_CANDIDATE_SCREEN']
    if any(r['status'] not in ('ARTIFICIAL_CANDIDATE_SCREEN', 'FAILED_EVALUATION') for r in rows):
        raise ValueError('Unknown utility state')
    criteria, blocks = {}, {}
    for key in scores.LIMITS:
        criteria[key] = sum(all(v[key] for v in r['checks'].values() if key in v) for r in good) / len(rows)
    for block in ('all', 'female', 'male', 'common'):
        blocks[block] = sum(all(r['checks'][block].values()) for r in good) / len(rows)
    losses, relationships = {}, {}
    if good:
        for group, metrics in {**good[0]['groups'], 'common': good[0]['common_audit']}.items():
            for key in metrics:
                values = [r['common_audit'][key] if group == 'common' else r['groups'][group][key] for r in good]
                losses[f'{group}/{key}'] = {'n_evaluated': len(values), 'mean': float(np.mean(values))}
        for key in good[0]['relationship_diagnostics']:
            values = [r['relationship_diagnostics'][key]['absolute_change'] for r in good]
            valid = [x for x in values if x is not None]
            relationships[key] = {'n_expected': len(rows), 'n_evaluated': len(valid),
                                   'mean': float(np.mean(valid)) if valid else None}
    return {'n_expected': len(rows), 'n_evaluated': len(good), 'failures': len(rows)-len(good),
            'all_passes': sum(r['all_passed'] for r in good),
            'all_pass_fraction': sum(r['all_passed'] for r in good) / len(rows),
            'criterion_pass_fractions': criteria, 'block_pass_fractions': blocks,
            'mean_losses': losses, 'relationship_diagnostics': relationships}


def privacy_summary(rows, level):
    valid = all(r['privacy'][level]['status'] == 'EVALUATED_NOT_A_PRIVACY_PASS' for r in rows)
    return {'n_expected': len(rows),
        'n_evaluated': sum(r['privacy'][level]['status'] == 'EVALUATED_NOT_A_PRIVACY_PASS' for r in rows),
        'means': {mode: {key: float(np.mean([r['privacy'][level][mode][key] for r in rows]))
                        for key in rows[0]['privacy'][level][mode]}
                  for mode in ('frozen', 'adapted')} if valid else None}


def identity():
    _, _, inherited = inputs()
    return {**inherited, **{str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT / 'joint_strength').glob('*.py'))}}


def run_cohort(model, cohort, output):
    started = time.monotonic()
    original, trained, _ = inputs()
    source_identity = identity()
    seeds = role_seeds(cohort)
    members = population.population(seeds['reference'], model)
    control = population.population(seeds['control'], model)
    audit = population.population(seeds['audit'], model)
    source_hashes = [fingerprint(x) for x in (members, control, audit)]
    if any(set(a['ids']) & set(b['ids']) for a, b in ((members, control), (members, audit), (control, audit))):
        raise ValueError('Role overlap')
    natural = {}
    for mode, base in (('independent', 115000000), ('bootstrap', 116000000)):
        rows = []
        for r in range(NATURAL_REPLICAS):
            seed = base + 1000 * cohort + r
            sample = population.population(seed, model) if mode == 'independent' else bootstrap(members, seed)
            rows.append({'sample_seed': seed, **safe_assess(members, sample, audit['z'])})
        natural[mode] = rows
    print(json.dumps({'model': model, 'cohort': cohort, 'phase': 'natural_complete'}), flush=True)
    negatives = [population.population(seeds['negative_base'] + r, model) for r in range(NEGATIVES)]
    negative_hashes = [fingerprint(x) for x in negatives]
    records = []
    for method in population.METHODS:
        for r in range(RELEASES + 1 + NEGATIVES):
            kind = 'cart' if r < RELEASES else 'copy_positive' if r == RELEASES else 'independent_negative'
            release = (synthesis.generate(members, method, 114000000 + 1000 * cohort + r)
                       if kind == 'cart' else members if kind == 'copy_positive' else negatives[r-RELEASES-1])
            release_hash = fingerprint(release)
            result = safe_assess(members, release, audit['z'])
            privacy = {level: privacy_evaluation(members, control, release, original, trained[model][level],
                       method, level, 118000000 + 1000 * cohort + r) for level in population.NOISE}
            if fingerprint(release) != release_hash:
                raise RuntimeError('Release mutated')
            records.append({'method': method, 'replica': r, 'kind': kind,
                            'release_sha256': release_hash, 'utility': result, 'privacy': privacy})
        print(json.dumps({'model': model, 'cohort': cohort, 'phase': method + '_complete'}), flush=True)
    if ([fingerprint(x) for x in (members, control, audit)] != source_hashes
            or [fingerprint(x) for x in negatives] != negative_hashes or identity() != source_identity):
        raise RuntimeError('Source or calculation input mutated')
    report = {'data_status': 'ARTIFICIAL_ONLY', 'norm_status': norms.NORM_STATUS,
              'model': model, 'cohort': cohort, 'seeds': seeds, 'limits': scores.LIMITS,
              'n_female': 197, 'n_male': 196, 'natural_replicas': NATURAL_REPLICAS,
              'releases': RELEASES, 'negative_releases': NEGATIVES,
              'source_sha256': source_identity, 'natural': natural, 'records': records}
    validate_cohort(report)
    path = Path(output) / 'cohorts' / f'{model}_{cohort}.json'
    write_report(report, path)
    summary = {'model': model, 'cohort': cohort,
               'natural': {mode: utility_summary(rows) for mode, rows in natural.items()}, 'methods': {}}
    for method in population.METHODS:
        summary['methods'][method] = {}
        for kind in ('cart', 'copy_positive', 'independent_negative'):
            rows = [r for r in records if r['method'] == method and r['kind'] == kind]
            passing = [r for r in rows if r['utility']['all_passed']]
            summary['methods'][method][kind] = {
                'utility': utility_summary([r['utility'] for r in rows]),
                'privacy': {level: privacy_summary(rows, level) for level in population.NOISE},
                'privacy_among_utility_passes': {level: privacy_summary(passing, level) if passing else None
                                                for level in population.NOISE}}
    summary['raw_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary['seconds'] = time.monotonic() - started
    write_report(summary, path.with_name(path.stem + '_summary.json'))
    return model, cohort, summary['seconds']


def validate_cohort(report):
    if report['limits'] != scores.LIMITS or report['data_status'] != 'ARTIFICIAL_ONLY':
        raise ValueError('Changed limits or data scope')
    if report['seeds'] != role_seeds(report['cohort']):
        raise ValueError('Changed seed schedule')
    if (report['n_female'], report['n_male']) != (197, 196):
        raise ValueError('Changed counts')
    if set(report['natural']) != {'independent', 'bootstrap'} or any(len(r) != NATURAL_REPLICAS for r in report['natural'].values()):
        raise ValueError('Incomplete natural design')
    expected = {(m, r) for m in population.METHODS for r in range(RELEASES+1+NEGATIVES)}
    seen = [(r['method'], r['replica']) for r in report['records']]
    if set(seen) != expected or len(seen) != len(expected):
        raise ValueError('Incomplete joint design')
    if any(set(r['privacy']) != set(population.NOISE) for r in report['records']):
        raise ValueError('Missing knowledge conditions')
    json.dumps(report, allow_nan=False)


def aggregate(output):
    output = Path(output)
    grouped = {}
    control_tests = []
    for model in population.MODELS:
        cohorts = [json.loads((output / 'cohorts' / f'{model}_{c}_summary.json').read_text()) for c in range(COHORTS)]
        if [r['cohort'] for r in cohorts] != list(range(COHORTS)):
            raise ValueError('Missing independent cohort')
        model_report = {'utility': {}, 'privacy': {}, 'paired_utility_differences': {}}
        sources = {mode: [c['natural'][mode] for c in cohorts] for mode in ('independent', 'bootstrap')}
        sources.update({m: [c['methods'][m]['cart']['utility'] for c in cohorts] for m in population.METHODS})
        for name, rows in sources.items():
            model_report['utility'][name] = {
                'intended': sum(r['n_expected'] for r in rows), 'failures': sum(r['failures'] for r in rows),
                'passes': sum(r['all_passes'] for r in rows),
                'all_pass': estimate([r['all_pass_fraction'] for r in rows], probability=True),
                'criteria': {k: estimate([r['criterion_pass_fractions'][k] for r in rows], probability=True)
                             for k in scores.LIMITS},
                'blocks': {k: estimate([r['block_pass_fractions'][k] for r in rows], probability=True)
                           for k in ('all', 'female', 'male', 'common')}}
        for method in population.METHODS:
            for comparator in ('independent', 'bootstrap'):
                differences = [a['all_pass_fraction']-b['all_pass_fraction']
                               for a, b in zip(sources[method], sources[comparator])]
                model_report['paired_utility_differences'][method + '_minus_' + comparator] = estimate(differences)
            for level in population.NOISE:
                for kind in ('cart', 'copy_positive', 'independent_negative'):
                    rows = [c['methods'][method][kind]['privacy'][level] for c in cohorts]
                    if any(r['means'] is None for r in rows):
                        model_report['privacy'][f'{method}/{level}/{kind}'] = {'status': 'NOT_EVALUABLE'}
                        continue
                    entry = {mode: {key: estimate([r['means'][mode][key] for r in rows])
                                    for key in rows[0]['means'][mode]} for mode in ('frozen', 'adapted')}
                    entry['paired_adapted_minus_frozen'] = {key: estimate([
                        r['means']['adapted'][key]-r['means']['frozen'][key] for r in rows]) for key in PRIMARY}
                    model_report['privacy'][f'{method}/{level}/{kind}'] = entry
                    if kind == 'independent_negative':
                        for mode in ('frozen', 'adapted'):
                            for key in PRIMARY:
                                values = np.asarray(entry[mode][key]['cohort_means'])
                                p = (1. if values[0] == 0 else 0.) if np.all(values == values[0]) else float(ttest_1samp(values, 0).pvalue)
                                control_tests.append({'model': model, 'method': method, 'knowledge': level,
                                    'attack': mode, 'endpoint': key, 'p': p})
        grouped[model] = model_report
    if len(control_tests) != 252:
        raise ValueError('Incomplete primary control family')
    for row, p in zip(control_tests, holm([r['p'] for r in control_tests])):
        row['holm_p'] = p
    result = {'data_status': 'ARTIFICIAL_ONLY', 'norm_status': norms.NORM_STATUS,
              'limits': scores.LIMITS, 'cohorts_per_model': COHORTS,
              'privacy_acceptance': 'NOT_DEFINED', 'joint_privacy_utility_pass_rate': None,
              'bootstrap_replicates': BOOTSTRAPS, 'models': grouped, 'control_tests': control_tests,
              'conditional_joint_results': 'Per-cohort privacy among utility-passing releases is retained in cohort summaries; empty sets are None.',
              'inference_limits': ['Ten independent reference/control pairs per model.',
                  'Shared seeds pair models; do not pool models as 30 independent replications.',
                  'Intervals conditional on fixed artificial mechanisms and saved shadow selection.',
                  'Pointwise t/bootstrap intervals are not simultaneous privacy acceptance bounds.',
                  'Zero bootstrap widths do not prove zero probability; cohort-any-pass exact intervals are a distinct endpoint.',
                  'Bootstrap cohorts are scientific stability controls, never privacy nonmembers.',
                  'No empirical anonymity, original-paper replication or calibrated risk threshold.']}
    write_report(result, output / 'results.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    if (args.output / 'results.json').exists() or list((args.output / 'cohorts').glob('*.json')):
        parser.error('Use an empty cohort output directory; no resume or overwrite')
    (args.output / 'cohorts').mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    before = identity()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_cohort, m, c, str(args.output)) for m in population.MODELS for c in range(COHORTS)]
        for future in as_completed(jobs):
            model, cohort, seconds = future.result()
            print(json.dumps({'completed': model, 'cohort': cohort, 'seconds': seconds}), flush=True)
    if before != identity():
        raise RuntimeError('Sources changed during study')
    aggregate(args.output)
    write_report({'status': 'COMPLETE', 'seconds': time.monotonic()-started,
                  'source_sha256': before, 'workers': args.workers,
                  'cart_releases': 6000, 'natural_comparisons': 6000,
                  'control_configuration_rows': 660,
                  'privacy_conditions': 19980}, args.output / 'COMPLETE.json')


if __name__ == '__main__':
    main()
