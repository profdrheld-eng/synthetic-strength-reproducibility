"""Local real-data reference helpers; no row export or network operations."""
import numpy as np
import pandas as pd
LIFTS=('SQ','BP','DL')
ZCOLS=('Z_squat','Z_benchpress','Z_deadlift')
RAW=('sq1rm','bp1rm','dl1rm')

def unique_pc1_weights(table):
    rows=table.dropna(subset=['z_score'])
    for _,group in rows.groupby('z_score'):
        if group.pc1_loading.nunique()!=1:raise ValueError('Conflicting repeated aggregate PC1 weight')
    return rows.drop_duplicates('z_score').set_index('z_score').loc[list(ZCOLS),'pc1_loading'].to_numpy(float)

def sexes(df):
    s=df['sex'].astype(str).str.strip().str.lower()
    out=np.where(s.str.startswith('männ'),'male',np.where(s.str.startswith('weib'),'female','unknown'))
    if 'unknown' in out:raise ValueError('Unknown sex category; no values logged')
    return out

def join_keys(z,t):
    for d,k in ((z,'code'),(t,'Probandencode')):
        if d[k].isna().any() or d[k].duplicated().any():raise ValueError('Missing or duplicated linkage key; no identifiers logged')
    return z.merge(t[['Probandencode']],left_on='code',right_on='Probandencode',how='inner',validate='one_to_one')

def design(df):
    age=pd.to_numeric(df.age,errors='coerce').to_numpy(float)
    height=pd.to_numeric(df.bodyheight,errors='coerce').to_numpy(float)
    mass=pd.to_numeric(df.bodymass,errors='coerce').to_numpy(float)
    logmass=np.full(len(df),np.nan);ok=np.isfinite(mass)&(mass>0);logmass[ok]=np.log(mass[ok])
    return np.column_stack([np.ones(len(df)),height,age,age**2,logmass,logmass**2])

def published_scores(df,constants):
    x=design(df);group=sexes(df);result=np.full((len(df),3),np.nan)
    for sex in ('female','male'):
        for j,(lift,raw) in enumerate(zip(LIFTS,RAW)):
            force=pd.to_numeric(df[raw],errors='coerce').to_numpy(float)
            valid=(group==sex)&np.isfinite(x).all(1)&np.isfinite(force)&(force>0)
            if not valid.any():continue
            c=constants[(sex,lift)]
            result[valid,j]=(np.log(force[valid])-x[valid]@c['coefficients'])/c['rsd']
    return result

def pca_svd(z):
    z=np.asarray(z,float)
    if not np.isfinite(z).all() or len(z)<3:raise ValueError('Finite complete profiles required')
    sd=z.std(0,ddof=1)
    if (sd<=0).any():raise ValueError('Degenerate PC1')
    u=(z-z.mean(0))/sd
    _,singular,vt=np.linalg.svd(u,full_matrices=False)
    w=vt[0]
    if w.sum()<0:w=-w
    raw=u@w
    return (raw-raw.mean())/raw.std(ddof=1),w,float(singular[0]**2/(singular**2).sum())
