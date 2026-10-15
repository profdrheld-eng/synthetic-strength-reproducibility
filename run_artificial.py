"""Reproduce the fixed artificial study; outputs stay outside this repository."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    if sys.flags.optimize:raise SystemExit('Run without -O; validation assertions are required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Plumbing check only, not paper reproduction')
    args = parser.parse_args()
    target = args.output.resolve()
    if target == ROOT or ROOT in target.parents:
        parser.error('Choose an output directory outside the repository')
    target.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    # Do not inherit any configuration pointing at protected empirical inputs.
    for key in list(env):
        if key.startswith('STRENGTH_'):
            del env[key]
    env.update(PYTHONPATH=str(ROOT/'source'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['STRENGTH_STUDY_OUTPUT'] = str(target/'study')
    env['STRENGTH_DIAGNOSTIC_OUTPUT'] = str(target/'diagnostics')
    def run(*command):
        subprocess.run([sys.executable, *map(str, command)], env=env, cwd=ROOT, check=True)
    if args.smoke:
        run('-m', 'joint_strength', '--cohorts', '1', '--releases', '1', '--output', target/'smoke.json')
        (target/'CHECK.json').write_text(json.dumps({'scope':'ARTIFICIAL_SMOKE_ONLY', 'passed':True})+'\n')
        return
    run('-m', 'joint_strength.study', '--output', target/'study')
    manifest = {'files': {str(p.relative_to(ROOT/'source')): {'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in sorted((ROOT/'source').rglob('*')) if p.is_file()}}
    (target/'study/SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    run(ROOT/'scripts/verify_artificial.py')
    run(ROOT/'diagnostics/diagnose.py')
    run(ROOT/'diagnostics/report.py')
    run(ROOT/'scripts/check_reported_values.py', '--output', target)

if __name__ == '__main__':
    main()
