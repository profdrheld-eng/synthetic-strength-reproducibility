"""Only artificial tests. Explicitly clear inherited protected-input configuration."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

def main():
    if sys.flags.optimize:raise SystemExit('Run without -O; validation assertions are required')
    root=Path(__file__).resolve().parent
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('STRENGTH_'):del env[key]
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for folder in ('source/tests','diagnostics','empirical','empirical_reference','tests'):
        env['PYTHONPATH']=os.pathsep.join((str(root/'source'),str(root/folder),str(root/'empirical')))
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/folder),'-p','test_*.py','-v'],cwd=root,env=env,check=True)
    with tempfile.TemporaryDirectory() as tmp:
        env['STRENGTH_EMPIRICAL_OUTPUT']=tmp
        subprocess.run([sys.executable,str(root/'empirical/verify_additional.py')],cwd=root,env=env,check=True)

if __name__=='__main__':main()
