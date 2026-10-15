"""Render exploratory diagnostic report and verify against aggregate source records."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from diagnose import ROOT, UPSTREAM, population, failure_keys, scores


def pct(x):
    return f'{100*x:.1f}'


def signed(x):
    return f'{x:+.3f}'


def interval(e):
    return f"{signed(e['mean'])} [{signed(e['bootstrap95'][0])}; {signed(e['bootstrap95'][1])}]"


def main():
    results = json.loads((ROOT / 'results.json').read_text())
    verified = {'cohorts': 0, 'tail_rows': 0, 'utility_flags': 0, 'max_aggregate_error': 0.}
    for path in sorted((ROOT / 'cohorts').glob('*.json')):
        c = json.loads(path.read_text())
        original_path = UPSTREAM / 'cohorts' / path.name
        assert hashlib.sha256(original_path.read_bytes()).hexdigest() == c['input_sha256']
        raw = json.loads(original_path.read_text())
        for arm, summary in c['arms'].items():
            utilities = raw['natural'][arm] if arm in raw['natural'] else [r['utility'] for r in raw['records'] if r['kind'] == 'cart' and r['method'] == arm]
            assert summary['passes'] == sum(u['all_passed'] for u in utilities)
            for key in scores.LIMITS:
                assert summary['failures'][key] == sum(any(not block[key] for block in u['checks'].values() if key in block) for u in utilities)
                assert summary['sole_failures'][key] == sum(failure_keys(u['checks']) == {key} for u in utilities)
            verified['utility_flags'] += sum(sum(len(b) for b in u['checks'].values()) for u in utilities)
            csvpath = ROOT / 'tails' / f"{c['model']}_{c['cohort']}_{arm}.csv"
            with csvpath.open() as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1800
            verified['tail_rows'] += len(rows)
            for key, values in summary['tails'].items():
                selected = [r for r in rows if r['key'] == key]
                assert len(selected) == 100
                for metric in ('reference', 'release', 'delta', 'absolute_change', 'exceeded'):
                    observations = [(r[metric] == 'True') if metric == 'exceeded' else float(r[metric]) for r in selected]
                    error = abs(float(np.mean(observations)) - values[metric])
                    verified['max_aggregate_error'] = max(verified['max_aggregate_error'], error)
                    assert error < 1e-12
        verified['cohorts'] += 1
    assert verified['tail_rows'] == 216000 and verified['utility_flags'] == 348000
    (ROOT / 'VERIFICATION.json').write_text(json.dumps(verified, indent=2) + '\n')
    text = ['# Gezielte Diagnostik: z-Score-Ränder und Beziehungen zu Körpermaßen', '',
        'Explorative Folgeauswertung ausschließlich künstlicher Daten. Drei Populationsmodelle, je zehn unabhängige Referenzkohorten mit 197 Frauen und 196 Männern. Je Kohorte 100 Freigaben für beide fixierten CART-Varianten sowie 100 unabhängige natürliche Stichproben und 100 geschichtete Bootstraps. Insgesamt 12.000 Vergleiche. Alle 13 Grenzen und 29 Einzelprüfungen bleiben unverändert. Methoden, Angriffe und Manuskript wurden nicht geändert.', '',
        '## Was tatsächlich zum Nichtbestehen führt', '',
        'Die z-Randprüfung verlangt, dass alle sechs Q05-/Q95-Werte der drei Übungen sowohl insgesamt als auch bei Frauen und Männern innerhalb von 0,20 z-Einheiten der Referenz bleiben. Die 18 Vergleiche sind abhängig. Die Spalte „nur z-Ränder“ zählt Freigaben, die ausschließlich diesen Kriterientyp verletzen; auch mehrere verletzte Randprüfungen zählen einmal. Dies ist eine Fehlerdiagnose, keine gelockerte Erhaltregel.', '',
        '| Modell | Verfahren/Kontrolle | alle Prüfungen bestanden /1000 | z-Ränder verletzt /1000 | nur z-Ränder verletzt /1000 | weitere häufige Verletzungen /1000 |',
        '|---|---|---:|---:|---:|---|']
    labels = {'independent': 'natürlich unabhängig', 'bootstrap': 'natürlicher Bootstrap', 'depth8_outcomes_first': 'CART Kraft zuerst', 'depth8_interleaved': 'CART abwechselnd'}
    export = []
    for model, arms in results['models'].items():
        for arm, a in arms.items():
            other = sorted([(v, k) for k, v in a['failures'].items() if k != 'z_tail_max'], reverse=True)[:3]
            text.append(f"| {model} | {labels[arm]} | {a['passes']} | {a['failures']['z_tail_max']} | {a['sole_failures']['z_tail_max']} | " + ', '.join(f'{k}: {v}' for v,k in other) + ' |')
            for kind in ('tails', 'relationships'):
                for key, metrics in a[kind].items():
                    export.append({'model': model, 'arm': arm, 'kind': kind, 'key': key,
                        **{metric: e['mean'] for metric, e in metrics.items()}})
    text += ['', 'Die z-Ränder sind ein häufiger Engpass, aber weitere Prüfungen scheitern gleichzeitig. Körpermaß-Korrelationen sind kein Bestandteil dieser Bestehensregel. `correlation_max` prüft die Korrelationen zwischen den drei z-Scores, nicht deren Beziehungen zu Alter, Größe oder Masse.', '',
        '## Welche Ränder betroffen sind', '',
        'Pro Modell und CART-Variante werden die drei Zellen mit der höchsten Überschreitungsquote gezeigt. Tabellenwerte sind Mittelwerte über zehn Kohorten; innerhalb einer Kohorte werden 100 Wiederholungen gemittelt. Q05 bezeichnet den unteren, Q95 den oberen Rand. Eine negative Q95-Verschiebung bedeutet einen niedrigeren oberen Rand. Eine positive Q05-Verschiebung bedeutet einen weniger negativen unteren Rand. Beide gemeinsam passen zu einer Einengung; einzelne Vorzeichen allein belegen keine globale Streuungsveränderung.', '',
        '| Modell | CART | Geschlecht/Übung/Rand | mittlere Verschiebung (z) | mittlerer absoluter Fehler (z) | >0,20 CART (%) | unabhängig (%) | Bootstrap (%) |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for model, arms in results['models'].items():
        for arm in population.METHODS:
            keys = sorted(arms[arm]['tails'], key=lambda k: arms[arm]['tails'][k]['exceeded']['mean'], reverse=True)[:3]
            for key in keys:
                a = arms[arm]['tails'][key]
                text.append(f"| {model} | {labels[arm]} | {key} | {signed(a['delta']['mean'])} | {a['absolute_change']['mean']:.3f} | {pct(a['exceeded']['mean'])} | {pct(arms['independent']['tails'][key]['exceeded']['mean'])} | {pct(arms['bootstrap']['tails'][key]['exceeded']['mean'])} |")
    text += ['', '### Gepaarte Randfehler gegenüber den Kontrollen', '',
        'Für die jeweils am häufigsten verletzte Zelle: Differenz des mittleren absoluten Randfehlers, CART minus Kontrolle, in z-Einheiten. Negative Werte bedeuten kleinere CART-Abweichungen. 95%-Intervalle aus 10.000 gepaarten Bootstraps der zehn Referenzkohorten.', '',
        '| Modell | CART | Zelle | minus unabhängig [95%] | minus Bootstrap [95%] |', '|---|---|---|---:|---:|']
    for model, arms in results['models'].items():
        for arm in population.METHODS:
            a = arms[arm]
            key = max(a['tails'], key=lambda k: a['tails'][k]['exceeded']['mean'])
            text.append(f"| {model} | {labels[arm]} | {key} | {interval(a['paired_differences']['independent']['tails'][key])} | {interval(a['paired_differences']['bootstrap']['tails'][key])} |")
    text += ['', '## Beziehungen zu Alter und Körpermaßen', '',
        'Ausgewertet wurden 48 bereits gespeicherte Pearson-/Spearman-Korrelationen (Frauen/Männer × SQ/BP/DL/PC1 × Alter/Größe/Masse × zwei Korrelationsarten). PC1 verwendet weiterhin die eingefrorene Referenztransformation. Zusätzlich zwölf deskriptive Pearson-Korrelationen der drei z-Scores mit den bereits definierten nichtlinearen Generatortermen. Es wurde keine neue Erhaltgrenze dafür festgelegt.', '',
        'Je Modell/Variante zeigt die nächste Tabelle die stärkste mittlere absolute Veränderung unter den 48 ursprünglichen Beziehungen. Die Auswahl ist explorativ.', '',
        '| Modell | CART | Beziehung | Referenz r | CART r | mittleres abs. Δr CART | unabhängig | Bootstrap | CART minus Bootstrap [95%] |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for nonlinear in (False, True):
        if nonlinear:
            text += ['', '### Bereits angelegte nichtlineare Beziehungen', '',
                'Der quadratische Altersterm ist `(a² − 1/3) × sqrt(45/4)` mit `a=(Alter−39)/21`. Die Größe-Masse-Wechselwirkung ist `h×m/sqrt(1+a²)` mit geschlechtsspezifischer Größenzentrierung und größenabhängiger Massenzentrierung, exakt nach dem eingefrorenen Generator (`source/strength_privacy/relationships.py`). Die Terme werden auf Referenz und Freigabe identisch berechnet. Sie sind gezielte Modell-Diagnosen, keine umfassende Messung aller nichtlinearen Beziehungen.', '',
                '| Modell | CART | Beziehung | Referenz r | CART r | mittleres abs. Δr CART | unabhängig | Bootstrap | CART minus Bootstrap [95%] |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
        for model, arms in results['models'].items():
            for arm in population.METHODS:
                a = arms[arm]
                keys = [k for k in a['relationships'] if ('age_quadratic' in k or 'height_mass_interaction' in k) == nonlinear]
                key = max(keys, key=lambda k: a['relationships'][k]['absolute_change']['mean'])
                v = a['relationships'][key]
                text.append(f"| {model} | {labels[arm]} | {key} | {v['reference']['mean']:.3f} | {v['release']['mean']:.3f} | {v['absolute_change']['mean']:.3f} | {arms['independent']['relationships'][key]['absolute_change']['mean']:.3f} | {arms['bootstrap']['relationships'][key]['absolute_change']['mean']:.3f} | {interval(a['paired_differences']['bootstrap']['relationships'][key])} |")
    text += ['', '### Alter gesondert', '',
        'Je Modell und CART-Variante die stärkste Veränderung unter den ursprünglichen Alter-Korrelationen und die stärkste Veränderung zum quadratischen Altersterm. Referenzwerte nahe null sind von einem tatsächlich starken Ausgangszusammenhang zu unterscheiden.', '',
        '| Modell | CART | Beziehung | Referenz r | CART r | mittleres abs. Δr CART | unabhängig | Bootstrap | CART minus Bootstrap [95%] |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for model, arms in results['models'].items():
        for arm in population.METHODS:
            a = arms[arm]
            for term in ('/age/', '/age_quadratic/'):
                keys = [k for k in a['relationships'] if term in k]
                key = max(keys, key=lambda k: a['relationships'][k]['absolute_change']['mean'])
                v = a['relationships'][key]
                text.append(f"| {model} | {labels[arm]} | {key} | {v['reference']['mean']:.3f} | {v['release']['mean']:.3f} | {v['absolute_change']['mean']:.3f} | {arms['independent']['relationships'][key]['absolute_change']['mean']:.3f} | {arms['bootstrap']['relationships'][key]['absolute_change']['mean']:.3f} | {interval(a['paired_differences']['bootstrap']['relationships'][key])} |")
    text += ['', '## Zentrale Befunde', '',
        'CART verletzt die gemeinsame z-Randprüfung in 90,9–96,3 % der Freigaben; unabhängige natürliche Stichproben in 98,5–99,0 %, natürliche Bootstraps in 89,5–95,0 %. Ausschließlich z-Ränder erklären bei CART lediglich 4–87 von 1.000 Freigaben je Variante/Modell. Die niedrige gemeinsame Passrate lässt sich deshalb nicht mit einem einzigen Randbereich erklären.', '',
        'Im ursprünglichen Modell und bei schweren Verteilungsrändern ist das obere Deadlift-Quantil der Frauen die häufigste einzelne CART-Randverletzung. Im verschobenen gemischten Modell sind beide Bankdrück-Ränder der Frauen auffällig. Die mittleren gerichteten Verschiebungen sind meist klein im Verhältnis zum mittleren absoluten Fehler. Die Kontrollen sprechen gegen eine allgemeine Erklärung durch systematische CART-Randverengung.', '',
        'Die deutlichere zusätzliche CART-Veränderung betrifft die nichtlineare Größe-Masse-Wechselwirkung beim Bankdrücken. Im ursprünglichen Modell sinkt deren Korrelation bei der jeweils am stärksten betroffenen Gruppe von 0,426 auf 0,270 (Kraft zuerst, Frauen) beziehungsweise von 0,452 auf 0,270 (abwechselnd, Männer). Mittlere absolute Änderungen von 0,158 beziehungsweise 0,184 liegen über den entsprechenden natürlichen Bootstraps (0,049 beziehungsweise 0,053). Beim quadratischen Altersterm zeigt Kraft zuerst im Heavy-Tails-Modell beim männlichen Kniebeugen-z-Score eine Abschwächung von 0,470 auf 0,367; hier beträgt die mittlere absolute Änderung 0,108 gegenüber 0,049 beim Bootstrap.', '',
        'Diese Beziehungen sind zusätzliche wissenschaftliche Verluststellen. Sie verursachen rechnerisch keine Verletzung der bisherigen Erhaltregel, weil diese keine Alters- oder Körpermaß-Beziehungsgrenze enthält. Eine kausale Zerlegung des CART-Mechanismus wurde nicht durchgeführt.', '',
        '## Einordnung und Grenzen', '',
        '- Natürliche unabhängige Stichproben haben eigene Stichprobenunsicherheit. Bootstraps verwenden dieselbe empirische Referenzverteilung und beantworten eine engere Frage. Beide Kontrollen bleiben getrennt.',
        '- Große absolute Korrelationsänderungen können bei schwachen Referenzkorrelationen auch zufällige Schwankungen darstellen. Für eine gezielte Einengung sprechen erst die Richtung der Veränderungen und der Vergleich mit den natürlichen Kontrollen.',
        '- Beziehungen zu Körpermaßen können sich verändern, auch wenn alle bisherigen Erhaltprüfungen bestehen. Die gemeinsame Erhaltprüfung beantwortet deshalb keine umfassende Frage zum Erhalt sämtlicher Beziehungen.',
        '- Die hier ausgewählten Maxima und ihre Intervalle sind explorativ und punktweise. Zehn Kohorten je Modell begrenzen die Sicherheit. Die drei Modelle werden nicht als 30 austauschbare unabhängige Kohorten zusammengefasst.',
        '- Keine neuen Datenschutzangriffe; diese Analyse erlaubt keine Aussage, dass Daten anonym sind. Die publizierte gerundete Norm bleibt unter der bisherigen Kennzeichnung vorläufig.', '',
        '## Reproduktion und Prüfungen', '',
        '`diagnose.py` nutzt ausschließlich den eingefrorenen künstlichen Generator und CART-Code des vorigen Laufs. `report.py` prüft CSV-Aggregate und Fehlerzählungen erneut gegen die gespeicherten Ausgangsergebnisse. `results.json` und `summary.csv` enthalten alle Zellen, nicht nur die ausgewählten Maxima. `cohorts/` enthält Kohortenaggregate; `tails/` enthält ausschließlich Quantile je künstlicher Freigabe, keine Personenzeilen.', '',
        'Ausführung aus dem Projektordner mit dem vorhandenen Interpreter in `outputs/strength_fresh_reproduction_2026-10-13/package/.venv/bin/python`, jeweils `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`: zuerst `-m unittest discover -s outputs/strength_tail_relationship_diagnostics_2026-10-14 -p test_diagnostics.py`, danach `outputs/strength_tail_relationship_diagnostics_2026-10-14/diagnose.py`, schließlich `outputs/strength_tail_relationship_diagnostics_2026-10-14/report.py`. Keine Installation erforderlich.', '',
        'Prüfnachweise: `CHECKS.json`, `VERIFICATION.json`, `tests.log`, `run.log`, `FILE_MANIFEST.json`. Unveränderte 32 eingefrorene Quelldateien; 6.000 CART-Fingerprints und neun z-Verteilungskennzahlen je Vergleich geprüft. 216.000 Randquantilzeilen und 348.000 gespeicherte Einzelprüfungen nachgezählt. Die Manifestdatei schließt sich selbst und Python-Cachedateien aus.', '',
        'Nächster empfohlener Schritt: Diese explorative Lokalisierung im gemeinsamen Studienprotokoll als Diagnostik festhalten. Die unveränderten primären Grenzen separat weiter berichten; eine neue Erhaltregel wäre eine eigene, vorab festzulegende Entscheidung.']
    (ROOT / 'REPORT.md').write_text('\n'.join(text) + '\n')
    fields = sorted(set(k for row in export for k in row))
    with (ROOT / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(export)
    files = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name != 'FILE_MANIFEST.json'}
    (ROOT / 'FILE_MANIFEST.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
    print(json.dumps(verified))


if __name__ == '__main__':
    main()
