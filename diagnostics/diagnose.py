"""Exploratory aggregate-only replay of frozen artificial study; no person input."""
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import os
import numpy as np

CODE = Path(__file__).resolve().parents[1] / 'source'
ROOT = Path(os.environ.get('STRENGTH_DIAGNOSTIC_OUTPUT', 'UNCONFIGURED_DIAGNOSTIC_OUTPUT'))
UPSTREAM = Path(os.environ.get('STRENGTH_STUDY_OUTPUT', 'UNCONFIGURED_STUDY_OUTPUT'))
sys.path.insert(0, str(CODE))
from strength_privacy import norms, population, synthesis, scores, relationships
from joint_strength.study import bootstrap, estimate
from joint_strength.runner import fingerprint


def failure_keys(checks):
    return {key for block in checks.values() for key, passed in block.items() if not passed}


def tail_rows(reference, release):
    rows = []
    for group in ('all', 'female', 'male'):
        arrays = []
        for data in (reference, release):
            mask = np.ones(len(data['sex']), bool) if group == 'all' else np.array(data['sex']) == group
            arrays.append(np.asarray(data['z'])[mask])
        for j, lift in enumerate(norms.LIFTS):
            qa, qb = [np.quantile(x[np.isfinite(x[:, j]), j], [.05, .95]) for x in arrays]
            for k, quantile in enumerate(('q05', 'q95')):
                delta = float(qb[k] - qa[k])
                rows.append({'key': f'{group}/{lift}/{quantile}', 'group': group,
                             'reference': float(qa[k]), 'release': float(qb[k]),
                             'delta': delta, 'absolute_change': abs(delta),
                             'exceeded': abs(delta) > scores.LIMITS['z_tail_max'] + 1e-12})
    return rows


def nonlinear_relationships(reference, release):
    """Existing generator terms, descriptive correlations only, no new gate."""
    result = {}
    feature_arrays = [relationships.features(d) for d in (reference, release)]
    for sex in ('female', 'male'):
        for j, lift in enumerate(norms.LIFTS):
            for k, name in ((3, 'age_quadratic'), (4, 'height_mass_interaction')):
                values = []
                for data, x in zip((reference, release), feature_arrays):
                    mask = np.array(data['sex']) == sex
                    values.append(float(np.corrcoef(x[mask, k], data['z'][mask, j])[0, 1]))
                result[f'{sex}/{lift}/{name}/pearson'] = {
                    'reference': values[0], 'release': values[1],
                    'absolute_change': abs(values[1] - values[0])}
    return result


def process(path):
    raw = json.loads(path.read_text())
    assert raw['limits'] == scores.LIMITS and raw['data_status'] == 'ARTIFICIAL_ONLY'
    model, cohort = raw['model'], raw['cohort']
    ref = population.population(raw['seeds']['reference'], model)
    assert sum(x == 'female' for x in ref['sex']) == 197
    assert sum(x == 'male' for x in ref['sex']) == 196
    arms = {mode: [(row, None) for row in rows] for mode, rows in raw['natural'].items()}
    for method in population.METHODS:
        arms[method] = [(r['utility'], r) for r in raw['records'] if r['kind'] == 'cart' and r['method'] == method]
    result = {'model': model, 'cohort': cohort, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'arms': {}}
    for arm, pairs in arms.items():
        out = {'n': len(pairs), 'passes': 0, 'failures': {}, 'sole_failures': {}, 'tails': {}, 'relationships': {}}
        tail_export = []
        max_error = 0.
        for utility, record in pairs:
            assert utility['status'] == 'ARTIFICIAL_CANDIDATE_SCREEN'
            if record is None:
                seed = utility['sample_seed']
                release = population.population(seed, model) if arm == 'independent' else bootstrap(ref, seed)
                replica = seed % 1000
            else:
                replica = record['replica']
                release = synthesis.generate(ref, arm, 114000000 + 1000 * cohort + replica)
                assert fingerprint(release) == record['release_sha256'], 'CART replay hash mismatch'
            rows = tail_rows(ref, release)
            for group in ('all', 'female', 'male'):
                a, b = [np.asarray(d['z'])[np.ones(len(d['sex']), bool) if group == 'all' else np.array(d['sex']) == group] for d in (ref, release)]
                actual = {'z_tail_max': max(r['absolute_change'] for r in rows if r['group'] == group),
                          'z_mean_max': float(np.max(abs(b.mean(0)-a.mean(0)))),
                          'z_sd_ratio_max': float(np.max(abs(b.std(0, ddof=1)/a.std(0, ddof=1)-1)))}
                for key, value in actual.items():
                    error = abs(value - utility['groups'][group][key])
                    max_error = max(max_error, error)
                    assert error < 1e-10, 'Natural/release distribution replay mismatch'
            failed = failure_keys(utility['checks'])
            assert utility['all_passed'] == (not failed)
            out['passes'] += int(not failed)
            for key in scores.LIMITS:
                out['failures'][key] = out['failures'].get(key, 0) + int(key in failed)
                out['sole_failures'][key] = out['sole_failures'].get(key, 0) + int(failed == {key})
            for r in rows:
                out['tails'].setdefault(r['key'], []).append(r)
                tail_export.append({'model': model, 'cohort': cohort, 'arm': arm, 'replica': replica, **r})
            diagnostic = {**utility['relationship_diagnostics'], **nonlinear_relationships(ref, release)}
            for key, r in diagnostic.items():
                assert r['reference'] is not None and r['release'] is not None
                d = r['release'] - r['reference']
                assert abs(abs(d)-r['absolute_change']) < 1e-12
                out['relationships'].setdefault(key, []).append({'reference': r['reference'], 'release': r['release'],
                    'delta': d, 'absolute_change': abs(d), 'failed_utility': bool(failed)})
        out['max_replay_error'] = max_error
        out['tails'] = {key: {metric: float(np.mean([r[metric] for r in rows]))
            for metric in ('reference', 'release', 'delta', 'absolute_change', 'exceeded')} for key, rows in out['tails'].items()}
        out['relationships'] = {key: {**{metric: float(np.mean([r[metric] for r in rows]))
            for metric in ('reference', 'release', 'delta', 'absolute_change')},
            'mean_abs_among_passes': float(np.mean([r['absolute_change'] for r in rows if not r['failed_utility']])) if any(not r['failed_utility'] for r in rows) else None}
            for key, rows in out['relationships'].items()}
        result['arms'][arm] = out
        target = ROOT / 'tails' / f'{model}_{cohort}_{arm}.csv'
        with target.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(tail_export[0]))
            writer.writeheader(); writer.writerows(tail_export)
    (ROOT / 'cohorts' / path.name).write_text(json.dumps(result, indent=2) + '\n')
    print(f'{model}/{cohort}: replay verified', flush=True)
    return result


def summarize(results):
    output = {'scope': 'ARTIFICIAL_ONLY_EXPLORATORY', 'limits': scores.LIMITS, 'models': {}}
    for model in population.MODELS:
        cohorts = sorted([r for r in results if r['model'] == model], key=lambda r: r['cohort'])
        assert len(cohorts) == 10
        model_result = {}
        for arm in cohorts[0]['arms']:
            rows = [r['arms'][arm] for r in cohorts]
            a = {'n': sum(r['n'] for r in rows), 'passes': sum(r['passes'] for r in rows),
                 'failures': {k: sum(r['failures'][k] for r in rows) for k in scores.LIMITS},
                 'sole_failures': {k: sum(r['sole_failures'][k] for r in rows) for k in scores.LIMITS}}
            for kind in ('tails', 'relationships'):
                metrics = ('reference', 'release', 'delta', 'absolute_change', 'exceeded') if kind == 'tails' else ('reference', 'release', 'delta', 'absolute_change')
                a[kind] = {key: {metric: estimate([r[kind][key][metric] for r in rows]) for metric in metrics} for key in rows[0][kind]}
            a['paired_differences'] = {}
            if arm in population.METHODS:
                for control in ('independent', 'bootstrap'):
                    a['paired_differences'][control] = {kind: {key: estimate([
                        r['arms'][arm][kind][key]['absolute_change'] - r['arms'][control][kind][key]['absolute_change'] for r in cohorts])
                        for key in rows[0][kind]} for kind in ('tails', 'relationships')}
            model_result[arm] = a
        output['models'][model] = model_result
    return output


def main():
    if not all(os.environ.get(k) for k in ('STRENGTH_DIAGNOSTIC_OUTPUT', 'STRENGTH_STUDY_OUTPUT')):
        raise SystemExit('Explicit study and diagnostic output paths required')
    ROOT.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((UPSTREAM / 'SOURCE_MANIFEST.json').read_text())['files']
    for name, meta in manifest.items():
        assert hashlib.sha256((CODE / name).read_bytes()).hexdigest() == meta['sha256']
    for directory in ('cohorts', 'tails'):
        (ROOT / directory).mkdir(exist_ok=True)
    paths = [UPSTREAM / 'cohorts' / f'{model}_{cohort}.json' for model in population.MODELS for cohort in range(10)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(process, paths))
    (ROOT / 'results.json').write_text(json.dumps(summarize(results), indent=2) + '\n')
    (ROOT / 'CHECKS.json').write_text(json.dumps({'source_files_verified': len(manifest), 'cohorts': len(results),
        'replayed_comparisons': 12000, 'cart_fingerprints_verified': 6000,
        'max_distribution_replay_error': max(a['max_replay_error'] for r in results for a in r['arms'].values()),
        'limits_unchanged': True, 'privacy_attacks_rerun': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
