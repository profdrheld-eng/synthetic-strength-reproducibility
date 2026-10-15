"""Private, aggregate-only empirical CART run. No network and no row files."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.umask(0o077)
import sys
import json,hashlib,time,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
import empirical as e
import numpy as np

def guard(event,args):
    if event=='socket.connect':raise PermissionError('Network disabled for empirical CART run')
sys.addaudithook(guard)
SCRIPT_ROOT=Path(__file__).resolve().parent
HERE=Path(os.environ.get('STRENGTH_EMPIRICAL_OUTPUT','UNCONFIGURED_EMPIRICAL_OUTPUT'))
DATA=Path(os.environ.get('STRENGTH_DATA','UNCONFIGURED_DATA'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')

def load():
    z=pd.read_excel(DATA,sheet_name='z_scores',usecols=['code','sex','age','bodymass','bodyheight','sq1rm','bp1rm','dl1rm','Z_squat','Z_benchpress','Z_deadlift'])
    t=pd.read_excel(DATA,sheet_name='TE',header=4,usecols=['Probandencode']).dropna()
    assert len(z)==393 and len(t)==393 and z.code.is_unique and t.Probandencode.is_unique
    linked=z.merge(t,left_on='code',right_on='Probandencode',how='inner',validate='one_to_one',sort=False)
    assert len(linked)==384
    zcols=['Z_squat','Z_benchpress','Z_deadlift'];a=linked.loc[linked[zcols].notna().all(axis=1)].copy()
    assert len(a)==374
    sex=a.sex.map({'weiblich':'female','männlich':'male'}).tolist()
    assert sex.count('female')==186 and sex.count('male')==188
    d={'ids':[f'LOCAL-EMP-{i}' for i in range(374)],'sex':sex,'age':a.age.to_numpy(float),'height':a.bodyheight.to_numpy(float),'mass':a.bodymass.to_numpy(float),'z':a[zcols].to_numpy(float),'strength':a[['sq1rm','bp1rm','dl1rm']].to_numpy(float)}
    e.validate(d);p=e.fit_pc1(d['z'])
    assert np.allclose(p.weights,[.5848693053561214,.5709168339762319,.5761786739673762],rtol=0,atol=1e-10)
    assert abs(p.variance_fraction-.8649694179132734)<1e-10
    return d

def controls(d,t,m,seed):
    pool=np.setdiff1d(t,m);rng=np.random.default_rng(seed);sx=np.array(d['sex']);indices=[]
    for sex in ('female','male'):
        candidates=pool[sx[pool]==sex];count=int(np.sum(sx[t]==sex))
        indices.extend(rng.choice(candidates,count,replace=True).tolist())
    assert not(set(indices)&set(m))
    a=e.subset(d,indices);a['ids']=[f'LOCAL-DISJOINT-DRAW-{seed}-{i}' for i in range(len(indices))]
    return a

def evaluate_case(d,t,h,method,rep,kind):
    base=1000*rep['fold']+rep['r'];m=e.balanced_members(d,t,h,143000000+base)
    training=e.subset(d,t);member=e.subset(d,m);control=e.subset(d,h)
    if kind=='cart':s=e.synthesize(training,method,142000000+base)
    elif kind=='copy':s=training
    else:s=controls(d,t,m,145000000+base)
    utility=e.assess(training,s,control['z'])
    levels=[];vectors=[]
    for level in e.LEVELS:
        v,supp=e.privacy_vectors(member,control,s,method,level,144000000+base)
        avg=e.standardized(v,member,control)
        assert np.isfinite(avg).all() and np.all((v>=0)&(v<=1))
        levels.append({'knowledge':level,'success':avg,'supplement':supp});vectors.append(v)
    row={'method':method,'fold':rep['fold'],'replica':rep['r'],'kind':kind,'training_n':len(t),'audit_n':len(h),'utility':utility,'privacy':levels}
    return row,np.array(vectors),m

def task(method,fold,fixture=False):
    d=synthetic() if fixture else load();n=len(d['ids']);folds=e.split_indices(d);h=folds[fold];t=np.setdiff1d(np.arange(n),h)
    num=np.zeros((3,2,2,7,n));den=np.zeros((2,n));rows=[]
    repetitions=1 if fixture else 100
    for r in range(repetitions):
        row,v,m=evaluate_case(d,t,h,method,{'fold':fold,'r':r},'cart');rows.append(row)
        for role,idx in enumerate((m,h)):
            for sex in e.WEIGHTS:
                mask=np.array(d['sex'])[idx]==sex;selected=idx[mask]
                # Per-person numerator is unweighted; person-role denominator counts observations.
                num[:,:,role][:,:,:,selected]+=v[:,:,role][:,:,:,mask]
                den[role,selected]+=1
        if not fixture and (r+1)%20==0:print(f'{method} fold={fold} releases={r+1}/100',flush=True)
    for k in range(1 if fixture else 11):
        kind='copy' if k==0 else 'source_excluded_control'
        row,_,_=evaluate_case(d,t,h,method,{'fold':fold,'r':100+k},kind);rows.append(row)
    return method,fold,rows,num,den

def synthetic():
    d=e.make_fixture(140000001,374);d['sex']=['female']*186+['male']*188;e.encode(d,d['z']);return d

def full_task(method):
    d=load();rows=[]
    for r in range(100):
        s=e.synthesize(d,method,146000000+r);rows.append({'method':method,'replica':r,'kind':'full_utility','utility':e.assess(d,s,d['z'])})
    print(f'{method} full-cohort utility complete',flush=True)
    return method,rows

def primary_estimate(num,den,sex):
    # Average each fold/release with equal weight, consistent with rows. Control sampling per fold differs slightly.
    # For bootstrap use stratum ratio of weighted contributions, with equal fold/release weight.
    result=np.zeros((3,2,2,7))
    sx=np.array(sex)
    for group,weight in e.WEIGHTS.items():
        mask=sx==group
        for role in range(2):result[:,:,role]+=weight*num[:,:,role][:,:,:,mask].sum(axis=3)/den[role,mask].sum()
    return result

def aggregate(rows,num,den,d):
    result={}
    for method in e.METHODS:
        own=[x for x in rows if x['method']==method];cart=[x for x in own if x['kind']=='cart'];full=[x for x in own if x['kind']=='full_utility']
        result[method]={'joint_releases':len(cart),'joint_utility_pass':sum(x['utility']['all_passed'] for x in cart),'full_releases':len(full),'full_utility_pass':sum(x['utility']['all_passed'] for x in full),'utility_checks':{},'privacy':{},'controls':{}}
        for group in cart[0]['utility']['checks']:
            for key in cart[0]['utility']['checks'][group]:
                name=f'{group}/{key}';result[method]['utility_checks'][name]={'joint_pass':sum(x['utility']['checks'][group][key] for x in cart),'full_pass':sum(x['utility']['checks'][group][key] for x in full)}
        avg=primary_estimate(num[method],den[method],d['sex'])
        for li,level in enumerate(e.LEVELS):
            ci=e.cluster_intervals(num[method][li],den[method],d['sex'])
            result[method]['privacy'][level]={}
            for mode,label in enumerate(('frozen','adapted')):
                result[method]['privacy'][level][label]={name:{'member':float(avg[li,mode,0,j]),'control':float(avg[li,mode,1,j]),'excess':float(avg[li,mode,0,j]-avg[li,mode,1,j]),'conditional_person_cluster_interval':ci[mode,j].tolist()} for j,name in enumerate(e.ENDPOINTS)}
        for kind in ('copy','source_excluded_control'):
            rr=[x for x in own if x['kind']==kind];result[method]['controls'][kind]={'releases':len(rr),'privacy':{}}
            for li,level in enumerate(e.LEVELS):
                av=np.mean([np.array(x['privacy'][li]['success']) for x in rr],axis=0)
                result[method]['controls'][kind]['privacy'][level]={label:{name:{'member':float(av[mode,0,j]),'control':float(av[mode,1,j]),'excess':float(av[mode,0,j]-av[mode,1,j])} for j,name in enumerate(e.ENDPOINTS)} for mode,label in enumerate(('frozen','adapted'))}
    return result

def main():
    if sys.flags.optimize:raise SystemExit('Run without -O; validation assertions are required')
    start=time.time();fixture='--fixture' in sys.argv
    required=['STRENGTH_PARAMETERS','STRENGTH_EMPIRICAL_OUTPUT']
    if not fixture:required+=['STRENGTH_DATA','STRENGTH_REFERENCE_OUTPUT']
    if not all(os.environ.get(k) for k in required):raise SystemExit('Missing explicit local input/output configuration')
    if SCRIPT_ROOT.parent.resolve() in HERE.resolve().parents or HERE.resolve()==SCRIPT_ROOT.parent.resolve():
        raise SystemExit('Choose an output directory outside the repository')
    HERE.mkdir(parents=True,exist_ok=False)
    if fixture:
        for method in e.METHODS:
            for fold in range(5):
                _,_,rows,num,den=task(method,fold,True)
                assert len(rows)==2 and rows[1]['utility']['all_passed']
                primary_estimate(num,den,synthetic()['sex'])
        save('SYNTHETIC_MIRROR.json',{'passed':True,'cohort_n':374,'fold_method_cases':10,'privacy_configurations':60,'row_export':False,'seconds':time.time()-start})
        print('Synthetic full-flow mirror passed',flush=True);return
    old=json.loads((Path(os.environ['STRENGTH_REFERENCE_OUTPUT'])/'INPUT_MANIFEST.json').read_text())['files']
    for name,expected in old.items():assert sha(Path(name))==expected, 'Previously audited input changed'
    frozen={str(p):sha(p) for p in [SCRIPT_ROOT/'empirical.py',SCRIPT_ROOT/'run.py',SCRIPT_ROOT/'test_empirical.py',*sorted(e.SOURCE.rglob('*.py')),*sorted((e.SOURCE/'fixtures').glob('privacy*parameters.json'))]}
    frozen.update(old);save('FROZEN_INPUTS.json',{'files':frozen,'limits':e.scores.LIMITS,'methods':e.synthesis.VARIANTS,'endpoints':e.ENDPOINTS,'knowledge':e.LEVELS,'bootstrap_repetitions':2000,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}})
    d=load();folds=e.split_indices(d)
    save('DESIGN.json',{'n_source':393,'n_linked':384,'n_complete':374,'female':186,'male':188,'folds':[{'fold':i,'training_n':374-len(h),'heldout_n':len(h),'heldout_female':sum(d['sex'][j]=='female' for j in h),'heldout_male':sum(d['sex'][j]=='male' for j in h),'split_sha256':hashlib.sha256(h.tobytes()).hexdigest()} for i,h in enumerate(folds)],'historical_pc1_variance':e.fit_pc1(d['z']).variance_fraction,'historical_z_max_error':float(np.max(abs((np.log(d['strength'])-e.eta(d))/e.scale(d)-d['z'])))})
    rows=[];num={m:np.zeros((3,2,2,7,374)) for m in e.METHODS};den={m:np.zeros((2,374)) for m in e.METHODS}
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(task,m,f) for m in e.METHODS for f in range(5)]
        for job in as_completed(jobs):
            method,fold,rr,nn,dd=job.result();rows.extend(rr)
            # Weight each person observation by the number of same-sex targets in that fold.
            h=folds[fold]
            for sex in e.WEIGHTS:
                mask=np.array(d['sex'])==sex;weight=1/sum(d['sex'][j]==sex for j in h)
                num[method][:,:,:,:,mask]+=nn[:,:,:,:,mask]*weight
                den[method][:,mask]+=dd[:,mask]*weight
            save(f'fold_{method}_{fold}.json',rr)
        fulljobs=[pool.submit(full_task,m) for m in e.METHODS]
        for job in as_completed(fulljobs):
            method,rr=job.result();rows.extend(rr);save(f'full_{method}.json',rr)
    assert len(rows)==1310
    for m in e.METHODS:
        assert np.all(den[m]>0)
        point=primary_estimate(num[m],den[m],d['sex'])
        naive=np.mean([np.array([p['success'] for p in r['privacy']]) for r in rows if r['method']==m and r['kind']=='cart'],axis=0)
        assert np.allclose(point,naive,rtol=0,atol=1e-12)
    summary=aggregate(rows,num,den,d);save('SUMMARY.json',summary)
    assert all(sha(Path(p))==h for p,h in frozen.items())
    save('RUN_COMPLETION.json',{'complete':True,'joint_releases':1000,'full_utility_releases':200,'control_releases':110,'paired_privacy_conditions':3330,'seconds':time.time()-start,'input_hashes_unchanged':True,'person_rows_exported':False,'network_disabled':True})
    print('Complete: 1,000 joint + 200 full utility + 110 controls; aggregate-only',flush=True)
if __name__=='__main__':main()
