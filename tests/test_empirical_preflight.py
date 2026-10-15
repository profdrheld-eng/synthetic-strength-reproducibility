"""Workflow regressions using invented parameters and non-dataset byte fixtures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
AUDIT_CHECKS=('pc1_diagnostics_passed','pc1_weights_and_correlations_passed',
              'historical_descriptives_passed','historical_z_validation_passed',
              'historical_z_reconstruction_passed','svd_check_passed','source_files_unchanged')

class EmpiricalPreflightTests(unittest.TestCase):
    def setUp(self):
        from artificial_parameters import write_parameters
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.params=self.root/'parameters';self.params.mkdir()
        write_parameters(self.params)
        self.data=self.root/'REGISTERED_FAKE_BYTES.xlsx';self.data.write_bytes(b'Not an Excel dataset. No person records.')
        self.audit=self.root/'audit';self.audit.mkdir()
        self.roles={'data':str(self.data),'coefficients':str(self.params/'coefficients.csv'),
                    'residual_scales':str(self.params/'residual_scales.csv')}
        report=self.audit/'AGGREGATE_RESULTS.json'
        report.write_text(json.dumps({'checks':dict.fromkeys(AUDIT_CHECKS,True)}))
        files={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in self.roles.values()}
        for i in range(7):
            p=self.root/f'other_audited_input_{i}.txt';p.write_text('Invented supporting input')
            files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        self.manifest={'schema_version':1,'roles':self.roles,'files':files,
                       'aggregate_results_sha256':hashlib.sha256(report.read_bytes()).hexdigest()}
        self.env=os.environ.copy()
        for key in list(self.env):
            if key.startswith('STRENGTH_'):del self.env[key]
        self.env.update(PYTHONPATH=str(ROOT/'empirical'),PYTHONDONTWRITEBYTECODE='1',
                        STRENGTH_PARAMETERS=str(self.params),STRENGTH_DATA=str(self.data),
                        STRENGTH_REFERENCE_OUTPUT=str(self.audit),STRENGTH_EMPIRICAL_OUTPUT=str(self.root/'output'))

    def execute(self,script):
        (self.audit/'INPUT_MANIFEST.json').write_text(json.dumps(self.manifest))
        return subprocess.run([sys.executable,'-c',script],env=self.env,cwd=ROOT,capture_output=True,text=True)

    def assert_rejected_before_data_loading(self):
        script="import run as r; r.load=lambda: (_ for _ in ()).throw(RuntimeError('UNVERIFIED_LOAD_REACHED')); r.main()"
        result=self.execute(script)
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('UNVERIFIED_LOAD_REACHED',result.stdout+result.stderr)
        self.assertFalse((self.root/'output/FROZEN_INPUTS.json').exists())

    def test_alternate_data_path_rejected_before_loading(self):
        self.env['STRENGTH_DATA']=str(self.root/'NEVER_CREATED_DATA.xlsx')
        self.assert_rejected_before_data_loading()

    def test_alternate_parameter_directory_rejected_before_loading(self):
        from artificial_parameters import write_parameters
        other=self.root/'other_parameters';other.mkdir();write_parameters(other)
        self.env['STRENGTH_PARAMETERS']=str(other)
        self.assert_rejected_before_data_loading()

    def test_empty_prior_manifest_rejected_before_loading(self):
        self.manifest['files']={}
        self.assert_rejected_before_data_loading()

    def test_failed_reference_audit_rejected_before_loading(self):
        report=self.audit/'AGGREGATE_RESULTS.json'
        checks=dict.fromkeys(AUDIT_CHECKS,True);checks['historical_z_reconstruction_passed']=False
        report.write_text(json.dumps({'checks':checks}))
        self.manifest['aggregate_results_sha256']=hashlib.sha256(report.read_bytes()).hexdigest()
        self.assert_rejected_before_data_loading()

    def test_changed_reference_report_rejected_before_loading(self):
        (self.audit/'AGGREGATE_RESULTS.json').write_text(json.dumps({'checks':dict.fromkeys(AUDIT_CHECKS,True),'changed':True}))
        self.assert_rejected_before_data_loading()

    def test_verified_artificial_input_binding_passes_without_parsing_data(self):
        result=self.execute("import run as r; files=r.validate_reference_inputs(); assert len(files)==12; print('PREFLIGHT_PASS')")
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('PREFLIGHT_PASS',result.stdout)

    def test_configured_adapter_import_does_not_open_parameters(self):
        self.env['STRENGTH_PARAMETERS']=str(self.root/'NEVER_CREATED_PARAMETERS')
        result=self.execute('import empirical; assert not empirical.COEFF and not empirical.SCALE')
        self.assertEqual(result.returncode,0,result.stderr)

    def reference_writer(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('reference_writer',ROOT/'empirical_reference/reproduce.py')
        sys.path.insert(0,str(ROOT/'empirical_reference'))
        try:
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            return module.write_audit_outputs
        finally:sys.path.pop(0)

    def test_failed_reference_writer_keeps_diagnostics_without_manifest(self):
        output=self.root/'failed_reference';output.mkdir()
        checks=dict.fromkeys(AUDIT_CHECKS,True);checks['svd_check_passed']=False
        with self.assertRaises(SystemExit):self.reference_writer()(output,{'checks':checks},self.manifest)
        self.assertTrue((output/'AGGREGATE_RESULTS.json').exists())
        self.assertFalse((output/'INPUT_MANIFEST.json').exists())

    def test_successful_reference_writer_binds_report_hash(self):
        output=self.root/'passed_reference';output.mkdir()
        self.reference_writer()(output,{'checks':dict.fromkeys(AUDIT_CHECKS,True)},self.manifest)
        manifest=json.loads((output/'INPUT_MANIFEST.json').read_text())
        self.assertEqual(manifest['aggregate_results_sha256'],hashlib.sha256((output/'AGGREGATE_RESULTS.json').read_bytes()).hexdigest())

    def test_spawned_worker_initializes_invented_reference(self):
        result=self.execute("""import multiprocessing as mp
import run as r
if __name__=='__main__':
    mp.set_start_method('spawn',force=True)
    with r.worker_pool() as pool:
        method,fold,rows,num,den=pool.submit(r.task,r.e.METHODS[0],0,True).result()
    assert fold==0 and len(rows)==2 and rows[1]['utility']['all_passed']
    print('SPAWN_FIXTURE_PASS')
""")
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('SPAWN_FIXTURE_PASS',result.stdout)

    def test_tie_entry_initializes_reference_before_loading_rows(self):
        (self.root/'output').mkdir()
        result=self.execute("""import verify_copy_ties as t
def intercept():
    assert len(t.e.COEFF)==6 and len(t.e.SCALE)==6
    print('TIE_CONFIGURED_BEFORE_ROWS')
    raise SystemExit(0)
t.r.load=intercept
t.main()
""")
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('TIE_CONFIGURED_BEFORE_ROWS',result.stdout)

    def test_empirical_verifier_does_not_require_mirror_artifact(self):
        # Synthetic aggregate contract only: no original Excel/model/parameter input.
        from artificial_parameters import write_parameters
        import empirical as e
        e.load_reference(self.params)
        utility=e.assess(e.make_fixture(201,80),e.make_fixture(201,80),e.make_fixture(202,30)['z'])
        output=self.root/'aggregate_verifier';output.mkdir()
        def save(name,value):(output/name).write_text(json.dumps(value))
        endpoints=e.ENDPOINTS;levels=e.LEVELS;methods=e.METHODS
        privacy=[{'knowledge':level,'success':[[[1.]*7]*2]*2} for level in levels]
        summary={}
        for method in methods:
            for fold in range(5):
                rows=[{'method':method,'fold':fold,'replica':i,'kind':('cart' if i<100 else 'copy' if i==100 else 'source_excluded_control'),
                       'utility':utility,'privacy':privacy} for i in range(111)]
                save(f'fold_{method}_{fold}.json',rows)
            save(f'full_{method}.json',[{'method':method,'replica':i,'kind':'full_utility','utility':utility} for i in range(100)])
            summary[method]={'joint_utility_pass':500,'full_utility_pass':100,
                'utility_checks':{f'{g}/{k}':{'joint_pass':500,'full_pass':100} for g,v in utility['checks'].items() for k in v},
                'privacy':{level:{mode:{name:{'member':1.,'control':1.,'excess':0.,'conditional_person_cluster_interval':[0.,0.]}
                                      for name in endpoints} for mode in ('frozen','adapted')} for level in levels}}
        save('FROZEN_INPUTS.json',{'files':self.manifest['files'],'limits':e.scores.LIMITS,'methods':methods,'knowledge':levels,'endpoints':endpoints})
        save('SUMMARY.json',summary);save('RUN_COMPLETION.json',{'complete':True,'input_hashes_unchanged':True})
        env=self.env.copy();env['STRENGTH_EMPIRICAL_OUTPUT']=str(output)
        result=subprocess.run([sys.executable,str(ROOT/'empirical/verify.py')],env=env,cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads((output/'VERIFICATION.json').read_text())['passed'])
        self.assertFalse((output/'SYNTHETIC_MIRROR.json').exists())

if __name__=='__main__':unittest.main()
