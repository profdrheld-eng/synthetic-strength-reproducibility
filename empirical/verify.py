"""Aggregate-only verification, no Excel/person data read."""
from pathlib import Path
import hashlib,json,os,sys
if sys.flags.optimize:raise SystemExit('Run without -O; validation assertions are required')
import numpy as np
HERE=Path(os.environ['STRENGTH_EMPIRICAL_OUTPUT'])
read=lambda name:json.loads((HERE/name).read_text())
frozen=read('FROZEN_INPUTS.json');summary=read('SUMMARY.json');completion=read('RUN_COMPLETION.json')
assert completion['complete'] and completion['input_hashes_unchanged']
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['files'].items())
limits=frozen['limits'];methods=frozen['methods'];checks=0;rows=[]
for method in methods:
    for fold in range(5):
        values=read(f'fold_{method}_{fold}.json');assert len(values)==111
        assert sum(x['kind']=='cart' for x in values)==100
        assert sum(x['kind']=='copy' for x in values)==1
        assert sum(x['kind']=='source_excluded_control' for x in values)==10
        assert len({(x['kind'],x['replica']) for x in values})==111
        rows.extend(values)
    full=read(f'full_{method}.json');assert len(full)==100;rows.extend(full)
for row in rows:
    u=row['utility'];expected={}
    for group,values in u['groups'].items():expected[group]={k:v<=limits[k]+1e-12 for k,v in values.items()}
    expected['common']={k:v<=limits[k]+1e-12 for k,v in u['common_audit'].items()}
    assert expected==u['checks'] and sum(map(len,expected.values()))==29
    assert u['all_passed']==all(all(v.values()) for v in expected.values())
    a,b=u['pc1_parameters']['reference'],u['pc1_parameters']['release']
    assert np.allclose(u['common_audit']['weight_max'],np.max(abs(np.array(a['weights'])-b['weights'])),rtol=0,atol=1e-12)
    assert abs(u['common_audit']['variance_fraction_change']-abs(a['variance_fraction']-b['variance_fraction']))<1e-12
    assert u['relationship_gate']=='NOT_APPLIED_NO_NEW_LIMITS'
    assert len(u['relationship_diagnostics'])==48
    if row['kind']=='copy':assert u['all_passed']
    for p in row.get('privacy',[]):
        v=np.array(p['success']);assert v.shape==(2,2,7) and np.isfinite(v).all() and np.all((v>=0)&(v<=1))
        # Equal body profiles create legitimate ties even for a verbatim copy.
        # Exact extended inference uses the other two lifts as well.
        if row['kind']=='copy' and p['knowledge']=='exact':assert np.all(v[0,0,4:]==1)
    checks+=29
assert len(rows)==1310
for method in methods:
    own=[x for x in rows if x['method']==method];cart=[x for x in own if x['kind']=='cart'];full=[x for x in own if x['kind']=='full_utility'];s=summary[method]
    assert s['joint_utility_pass']==sum(x['utility']['all_passed'] for x in cart)
    assert s['full_utility_pass']==sum(x['utility']['all_passed'] for x in full)
    for name,value in s['utility_checks'].items():
        group,key=name.split('/')
        assert value['joint_pass']==sum(x['utility']['checks'][group][key] for x in cart)
        assert value['full_pass']==sum(x['utility']['checks'][group][key] for x in full)
    for li,level in enumerate(frozen['knowledge']):
        av=np.mean([np.array(x['privacy'][li]['success']) for x in cart],axis=0)
        for mode,label in enumerate(('frozen','adapted')):
            for j,name in enumerate(frozen['endpoints']):
                value=s['privacy'][level][label][name]
                assert abs(value['member']-av[mode,0,j])<1e-12 and abs(value['control']-av[mode,1,j])<1e-12
                assert abs(value['excess']-(value['member']-value['control']))<1e-12
                assert value['conditional_person_cluster_interval'][0]<=value['conditional_person_cluster_interval'][1]
assert sum(len(x.get('privacy',[])) for x in rows)==3330
assert read('SYNTHETIC_MIRROR.json')['passed']
(HERE/'VERIFICATION.json').write_text(json.dumps({'passed':True,'utility_rows':len(rows),'threshold_decisions_recomputed':checks,'paired_privacy_conditions':3330,'privacy_summary_recomputed':True,'copy_control_exact_frozen_extended_inference_member_scores_one':True,'copy_control_demographic_ties_not_assumed_absent':True,'input_hashes_unchanged':True,'source_person_data_read_by_verifier':False},indent=2)+'\n')
print(f'PASS: {len(rows)} utility records, {checks} unchanged threshold decisions, 3330 paired privacy conditions')
