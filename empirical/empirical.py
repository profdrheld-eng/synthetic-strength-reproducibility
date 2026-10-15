"""Private empirical adapter; fixed scientific kernels, historical reference only."""
from pathlib import Path
import ast,csv,hashlib,json,sys
from types import SimpleNamespace
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'source'
sys.path.insert(0,str(SOURCE))
from strength_privacy import attacks,adaptive,synthesis,scores,population
from joint_strength.pc1 import fit_pc1
LIFTS=('SQ','BP','DL');METHODS=tuple(synthesis.VARIANTS)
LEVELS=('exact','mild','strong');WEIGHTS={'female':186/374,'male':188/374}
COEFF={};SCALE={}

def load_reference(directory):
    """Load explicitly supplied aggregate parameters, never participant rows/models."""
    directory=Path(directory);coeff={};scale={}
    with (directory/'coefficients.csv').open() as f:
        for row in csv.DictReader(f):
            if row['variant']=='reduced':
                key=(row['stratum'],row['lift']);term=row['term']
                if term in coeff.setdefault(key,{}):raise ValueError('Duplicate coefficient')
                coeff[key][term]=float(row['coefficient'])
    with (directory/'residual_scales.csv').open() as f:
        for row in csv.DictReader(f):
            if row['variant']=='reduced':
                key=(row['stratum'],row['lift'])
                if key in scale:raise ValueError('Duplicate residual scale')
                scale[key]=float(row['sd_residuals'])
    keys={(sex,lift) for sex in ('female','male') for lift in LIFTS}
    terms={'intercept','height_cm','age_years','age_squared','log_mass_kg','log_mass_squared'}
    if set(coeff)!=keys or set(scale)!=keys:raise ValueError('Incomplete reference')
    if any(set(c)!=terms or not np.isfinite(list(c.values())).all() for c in coeff.values()):raise ValueError('Invalid coefficient schema/value')
    if any(not np.isfinite(s) or s<=0 for s in scale.values()):raise ValueError('Invalid residual scale')
    COEFF.clear();COEFF.update(coeff);SCALE.clear();SCALE.update(scale)

ORIGINAL=json.loads((SOURCE/'fixtures/privacy_shadow_parameters.json').read_text())
ADAPTED=json.loads((SOURCE/'fixtures/privacy_adaptive_shadow_parameters.json').read_text())['original']

def subset(d,indices):
    indices=np.asarray(indices,dtype=int)
    return {k:[v[i] for i in indices] if isinstance(v,list) else v[indices].copy() for k,v in d.items()}

def eta(d):
    lm=np.log(d['mass']);x=np.column_stack([np.ones(len(lm)),d['height'],d['age'],d['age']**2,lm,lm**2]);out=np.empty((len(lm),3))
    for sex in ('female','male'):
        mask=np.array(d['sex'])==sex
        for j,lift in enumerate(LIFTS):out[mask,j]=x[mask]@np.array([COEFF[(sex,lift)][key] for key in ('intercept','height_cm','age_years','age_squared','log_mass_kg','log_mass_squared')])
    return out

def scale(d):return np.array([[SCALE[(s,l)] for l in LIFTS] for s in d['sex']])

def encode(d,z):
    d['z']=np.asarray(z,float).copy();d['strength']=np.exp(eta(d)+scale(d)*d['z']);return d

def validate(d):
    n=len(d['ids'])
    if len(set(d['ids']))!=n or len(d['sex'])!=n or set(d['sex'])!={'female','male'}:raise ValueError('Invalid internal identity/sex schema')
    for key in ('age','height','mass'):
        if np.shape(d[key])!=(n,) or not np.isfinite(d[key]).all() or np.any(d[key]<=0):raise ValueError('Invalid body variables')
    if np.shape(d['z'])!=(n,3) or np.shape(d['strength'])!=(n,3) or not np.isfinite(d['z']).all() or not np.isfinite(d['strength']).all() or (d['strength']<=0).any():raise ValueError('Complete positive lift profiles required')
    if not np.allclose((np.log(d['strength'])-eta(d))/scale(d),d['z'],atol=1e-10,rtol=0):raise ValueError('Historical score/raw mismatch')

# All 29 scientific checks are the exact original function bodies; only data validation/norm reference differ.
path=SOURCE/'joint_strength/utility.py';tree=ast.parse(path.read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('distribution','correlation','relationships','assess')]
namespace={'np':np,'rankdata':rankdata,'norms':SimpleNamespace(LIFTS=LIFTS),'LIMITS':scores.LIMITS,'fit_pc1':fit_pc1,'validate_data':validate}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
def assess(a,b,audit):
    result=namespace['assess'](a,b,audit)
    # The original marker describes the old artificial runner, not these empirical data.
    result['status']='EMPIRICAL_CONDITIONAL_UTILITY_SCREEN'
    return result

def synthesize(reference,method,seed):
    cfg=synthesis.VARIANTS[method];order=np.array(cfg['order'])
    x=np.column_stack([reference['age'],reference['height'],reference['mass'],reference['z']])
    sampled=synthesis.sample_matrix(x[:,order],np.array(reference['sex']),seed,cfg)[:,np.argsort(order)]
    d={'ids':[f'LOCAL-SYNTH-{seed}-{i}' for i in range(len(x))],'sex':reference['sex'].copy(),'age':sampled[:,0],'height':sampled[:,1],'mass':sampled[:,2]}
    encode(d,sampled[:,3:]);validate(d);return d

def make_fixture(seed,n):
    rng=np.random.default_rng(seed);sex=['female' if i%2==0 else 'male' for i in range(n)]
    d={'ids':[f'TEST-{seed}-{i}' for i in range(n)],'sex':sex,'age':rng.uniform(18,60,n),'height':rng.uniform(155,195,n),'mass':rng.uniform(50,100,n)}
    z=rng.normal(size=(n,1))+.4*rng.normal(size=(n,3));return encode(d,z)

def split_indices(d,seed=141000000):
    rng=np.random.default_rng(seed);folds=[[] for _ in range(5)]
    for sex in ('female','male'):
        values=np.flatnonzero(np.array(d['sex'])==sex);rng.shuffle(values)
        for i,part in enumerate(np.array_split(values,5)):folds[i].extend(part.tolist())
    return [np.array(sorted(x),int) for x in folds]

def balanced_members(d,training,heldout,seed):
    rng=np.random.default_rng(seed);out=[];sx=np.array(d['sex'])
    for sex in ('female','male'):
        n=int(np.sum(sx[heldout]==sex));pool=training[sx[training]==sex]
        if len(pool)<n or n<2:raise ValueError('Insufficient equal candidate sets')
        out.extend(rng.choice(pool,n,replace=False).tolist())
    return np.array(out,int)

def bridge_scores(a,b,release,scale_a,scale_b,k,seed):
    order=np.random.default_rng(seed).permutation(len(b));truth=np.argsort(order)
    wa=attacks.neighbours(a,release[:,:3],np.asarray(scale_a),k)
    wb=attacks.neighbours(b[order],release[:,3:],np.asarray(scale_b),k)
    values=wa@wb.T;maxima=np.isclose(values,values.max(1)[:,None],rtol=0,atol=1e-12)
    return maxima[np.arange(len(a)),truth]/maxima.sum(1)

def observations(d,level,seed):
    age,height,mass,force=population.NOISE[level]
    return attacks.matrix(d)+np.random.default_rng(seed).normal(size=(len(d['ids']),6))*np.array([age,height,mass,force,force,force])

ENDPOINTS=('link',*[f'inference_{p}_{lift}' for p in ('demographics','extended') for lift in LIFTS])
def privacy_vectors(m,c,syn,method,level,seed):
    attacks.check_disjoint(m,c)
    for sex in ('female','male'):
        if m['sex'].count(sex)!=c['sex'].count(sex):raise ValueError('Unequal attack candidates')
    obs=[observations(d,level,seed+i*1000000) for i,d in enumerate((m,c))]
    truth=[attacks.matrix(d) for d in (m,c)];release=attacks.matrix(syn)
    vectors=np.empty((2,2,7,len(m['ids'])));supplement={}
    chosen=ADAPTED[level]['selected'][method]
    for sex in ('female','male'):
        masks=[np.array(d['sex'])==sex for d in (m,c,syn)];s=release[masks[2]]
        frozen_scale=np.array(ORIGINAL['scales'][sex]);adapt_scale=np.array(ADAPTED[level]['scales'][sex])
        parameters=ORIGINAL['parameters'][method]
        for mode in range(2):
            for role in range(2):
                t=truth[role][masks[role]];o=obs[role][masks[role]]
                kind='frozen' if mode==0 else chosen['link']
                sc=frozen_scale if kind=='frozen' else adapt_scale
                k=parameters['link_k'] if kind=='frozen' else (5 if kind=='trees' else int(kind[2:]))
                vectors[mode,role,0,masks[role]]=bridge_scores(o[:,:3],o[:,3:],s,sc[:3],sc[3:],k,seed)
                for j,family in enumerate(ENDPOINTS[1:],1):
                    _,profile,lift=family.split('_');secret=3+LIFTS.index(lift);enhanced=profile=='extended'
                    kind='frozen' if mode==0 else chosen[family]
                    if kind=='frozen':prediction=attacks.predict(o,s,secret,parameters['inference_k'],frozen_scale,enhanced)
                    else:prediction=adaptive.infer(o,s,secret,enhanced,adapt_scale,kind,seed)
                    vectors[mode,role,j,masks[role]]=abs(np.expm1(prediction-t[:,secret]))<=.05
        width=attacks.WIDTHS[parameters['singling_width']]
        for role,label in enumerate(('member','control')):
            value=attacks.predicates(s,truth[role][masks[role]],width)
            supplement[f'{sex}/singling_{label}']=value['success_rate']
            supplement[f'{sex}/singling_queries']=value['queries']
            supplement[f'{sex}/aux_verified_{label}']=population.auxiliary_confirmation(s,truth[role][masks[role]],obs[role][masks[role]],width)
    return vectors,supplement

def standardized(vectors,m,c):
    out=np.zeros((2,2,7))
    for role,d in enumerate((m,c)):
        for sex in WEIGHTS:out[:,role]+=WEIGHTS[sex]*vectors[:,role][:,:,np.array(d['sex'])==sex].mean(axis=2)
    return out

# Same individual across roles remains coupled. No refitting or claim of population calibration.
def cluster_intervals(num,den,sex,seed=148000000):
    rng=np.random.default_rng(seed);estimates=np.zeros((2000,2,2,7))
    for group,weight in WEIGHTS.items():
        idx=np.flatnonzero(np.array(sex)==group);draw=idx[rng.integers(len(idx),size=(2000,len(idx)))]
        for mode in range(2):
            for role in range(2):
                denominator=den[role,draw].sum(1)
                if np.any(denominator<=0):raise ValueError('Unobserved person-role bootstrap stratum')
                estimates[:,mode,role]+=weight*num[mode,role,:,draw].sum(axis=1)/denominator[:,None]
    return np.quantile(estimates[:,:,0]-estimates[:,:,1],[.025,.975],axis=0).transpose(1,2,0)
