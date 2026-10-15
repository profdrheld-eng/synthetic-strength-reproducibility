import csv
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class PackagingTests(unittest.TestCase):
    def test_adapter_import_has_no_historical_parameter_dependency(self):
        env=os.environ.copy()
        env['PYTHONPATH']=str(ROOT/'empirical')
        for key in list(env):
            if key.startswith('STRENGTH_'):del env[key]
        subprocess.run([sys.executable,'-c','import empirical; assert not empirical.COEFF and not empirical.SCALE'],env=env,check=True,cwd=ROOT)

    def test_reference_import_does_not_start_empirical_analysis(self):
        spec=importlib.util.spec_from_file_location('reference_reproduction',ROOT/'empirical_reference/reproduce.py')
        sys.path.insert(0,str(ROOT/'empirical_reference'))
        try:
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.assertTrue(callable(module.main))
        finally:sys.path.pop(0)

    def test_invalid_reference_is_rejected_without_state_change(self):
        import empirical as e
        from artificial_parameters import write_parameters
        with tempfile.TemporaryDirectory() as tmp:
            write_parameters(tmp);e.load_reference(tmp)
            before=(dict(e.COEFF),dict(e.SCALE))
            path=Path(tmp)/'residual_scales.csv'
            with path.open() as f:rows=list(csv.DictReader(f))
            for value in ('0','-1','nan','inf'):
                rows[0]['sd_residuals']=value
                with path.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
                with self.assertRaises(ValueError):e.load_reference(tmp)
                self.assertEqual((e.COEFF,e.SCALE),before)

    def test_duplicate_coefficient_rejected(self):
        import empirical as e
        from artificial_parameters import write_parameters
        with tempfile.TemporaryDirectory() as tmp:
            write_parameters(tmp)
            path=Path(tmp)/'coefficients.csv'
            with path.open() as f:rows=f.readlines()
            with path.open('a') as f:f.write(rows[1])
            with self.assertRaises(ValueError):e.load_reference(tmp)

    def test_outputs_cannot_be_written_into_repository(self):
        completed=subprocess.run([sys.executable,str(ROOT/'run_artificial.py'),'--smoke','--output',str(ROOT/'forbidden_output')],capture_output=True,text=True)
        self.assertNotEqual(completed.returncode,0)
        self.assertIn('outside the repository',completed.stderr)
        self.assertFalse((ROOT/'forbidden_output').exists())

    def test_empirical_command_requires_explicit_inputs(self):
        env=os.environ.copy()
        for key in list(env):
            if key.startswith('STRENGTH_'):del env[key]
        result=subprocess.run([sys.executable,str(ROOT/'empirical/run.py')],env=env,capture_output=True,text=True,cwd=ROOT)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Missing explicit local input/output configuration',result.stderr)

    def test_full_flow_mirror_uses_only_invented_parameters(self):
        import json
        from artificial_parameters import write_parameters
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);parameters=tmp/'parameters';parameters.mkdir()
            write_parameters(parameters)
            env=os.environ.copy()
            for key in list(env):
                if key.startswith('STRENGTH_'):del env[key]
            env.update(STRENGTH_PARAMETERS=str(parameters),STRENGTH_EMPIRICAL_OUTPUT=str(tmp/'mirror'),
                       STRENGTH_DATA=str(tmp/'ABSENT_PERSON_DATA.xlsx'))
            subprocess.run([sys.executable,str(ROOT/'empirical/run.py'),'--fixture'],env=env,check=True,cwd=ROOT,capture_output=True)
            output=tmp/'mirror'
            result=json.loads((output/'SYNTHETIC_MIRROR.json').read_text())
            self.assertTrue(result['passed'])
            self.assertFalse(result['row_export'])
            self.assertEqual(result['fold_method_cases'],10)
            self.assertEqual([p.name for p in output.iterdir()],['SYNTHETIC_MIRROR.json'])

    def test_inventory_scanner_does_not_exempt_its_own_file(self):
        import hashlib
        import json
        spec=importlib.util.spec_from_file_location('inventory_check',ROOT/'scripts/verify_repository.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);(tmp/'scripts').mkdir()
            name='scripts/verify_repository.py'
            raw=('/'+'Users/ARTIFICIAL_TEST_ACCOUNT/forbidden').encode()
            (tmp/name).write_bytes(raw)
            (tmp/'FILES.json').write_text(json.dumps({'files':{name:{'sha256':hashlib.sha256(raw).hexdigest()}}}))
            subprocess.run(['git','init','-q',str(tmp)],check=True)
            subprocess.run(['git','add','--',name,'FILES.json'],cwd=tmp,check=True)
            module.ROOT=tmp
            with self.assertRaises(AssertionError):module.main()

if __name__=='__main__':unittest.main()
