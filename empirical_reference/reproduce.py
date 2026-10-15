"""Authorised local empirical audit. Aggregate-only outputs, no synthesis."""
from pathlib import Path
import ast,csv,hashlib,json,os,sys,platform
import numpy as np
import pandas as pd
from scipy import stats
from reference_utils import *
def main():
    if sys.flags.optimize:raise SystemExit('Run without -O; validation assertions are required')
    if not all(os.environ.get(k) for k in ('STRENGTH_REFERENCE_OUTPUT','STRENGTH_ORIGINAL_PROJECT','STRENGTH_PARAMETERS')):
        raise SystemExit('Explicit original-project, parameter and output paths required')
    os.umask(0o077)
    HERE=Path(os.environ['STRENGTH_REFERENCE_OUTPUT'])
    repo=Path(__file__).resolve().parents[1]
    if HERE.resolve()==repo or repo in HERE.resolve().parents:raise SystemExit('Choose an output directory outside the repository')
    HERE.mkdir(parents=True,exist_ok=False)
    PROJECT=Path(os.environ['STRENGTH_ORIGINAL_PROJECT'])
    PARAMETERS=Path(os.environ['STRENGTH_PARAMETERS'])
    SOURCE=Path(__file__).resolve().parents[1]/'source'
    # Deny network connections while empirical data are in memory.
    def guard(event,args):
        if event=='socket.connect':raise PermissionError('Network disabled for empirical reference run')
    sys.addaudithook(guard)
    paths=[PROJECT/'df_updated.xlsx',PROJECT/'analysis_pipeline.py',PROJECT/'tools/run_pc1_comparison.py',
           PROJECT/'results_pc1_final/00_pc1_diagnostics.csv',PROJECT/'results_pc1_final/00_pc1_loadings_and_correlations.csv',
           PROJECT/'results_pc1_final/01_sample_descriptives.csv',PROJECT/'results_pc1_final/02_zscore_validation.csv',
           SOURCE/'fixtures/S1_reference_constants.csv',
           PARAMETERS/'coefficients.csv',PARAMETERS/'residual_scales.csv']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p):sha(p) for p in paths}
    z=pd.read_excel(paths[0],sheet_name='z_scores',usecols=['id','code','sex','age','bodymass','bodyheight',*RAW,*ZCOLS])
    t=pd.read_excel(paths[0],sheet_name='TE',header=4,usecols=['Probandencode']).dropna(subset=['Probandencode'])
    linked=join_keys(z,t)
    sex=sexes(z);ls=sexes(linked)
    complete=linked[list(ZCOLS)].apply(pd.to_numeric,errors='coerce').notna().all(1)
    zarray=z[list(ZCOLS)].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    assert np.isfinite(zarray[~np.isnan(zarray)]).all()
    constants={}
    with paths[7].open() as f:
        for row in csv.DictReader(f):
            constants[(row['stratum'],row['lift'])]={'coefficients':[float(row[k]) for k in ('intercept','height_cm','age_years','age_squared','log_mass_kg','log_mass_squared')],'rsd':float(row['residual_scale'])}
    formula=published_scores(z,constants)
    flow={'reference':{'n':len(z),'female':int((sex=='female').sum()),'male':int((sex=='male').sum())},
          'linked':{'n':len(linked),'female':int((ls=='female').sum()),'male':int((ls=='male').sum())},
          'pc1_complete_linked':{'n':int(complete.sum()),'female':int(((ls=='female')&complete).sum()),'male':int(((ls=='male')&complete).sum())},
          'complete_in_full_393':int(np.isfinite(zarray).all(1).sum()),'training_rows_with_keys':len(t),
          'missing_reference_keys':int(z.code.isna().sum()),'duplicate_reference_keys':int(z.code.duplicated().sum()),
          'duplicate_training_keys':int(t.Probandencode.duplicated().sum()),
          'unmatched_reference_keys':int((~z.code.isin(t.Probandencode)).sum()),
          'unmatched_training_keys':int((~t.Probandencode.isin(z.code)).sum()),
          'matches_after_strip_casefold_diagnostic_only':int(z.code.astype(str).str.strip().str.casefold().isin(t.Probandencode.astype(str).str.strip().str.casefold()).sum())}
    input_quality={}
    for column in ('age','bodymass','bodyheight',*RAW,*ZCOLS):
        numeric=pd.to_numeric(z[column],errors='coerce').to_numpy(float)
        input_quality[column]={'missing':int(np.isnan(numeric).sum()),'infinite':int(np.isinf(numeric).sum()),
            'unparseable_nonmissing':int((z[column].notna()&pd.to_numeric(z[column],errors='coerce').isna()).sum()),
            'zero':int((numeric==0).sum()),'negative':int((numeric<0).sum())}
    flow['duplicate_id_field']=int(z.id.duplicated().sum())
    # Execute only the exact documented PC1 function AST, never the original whole pipeline.
    tree=ast.parse(paths[2].read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_pc1')
    namespace={'np':np,'pd':pd}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(paths[2]),'exec'),namespace)
    original_function_sha=hashlib.sha256(ast.get_source_segment(paths[2].read_text(),fn).encode()).hexdigest()
    computed,diagnostics,loadings=namespace['compute_pc1'](linked)
    expected=pd.read_csv(paths[3]);dcompare=[]
    for row in expected.itertuples():
        actual=float(diagnostics.loc[diagnostics.metric==row.metric,'value'].iloc[0]);error=abs(actual-float(row.value))
        dcompare.append({'metric':row.metric,'expected':float(row.value),'reproduced':actual,'absolute_error':error,'passed':error<=1e-10})
    expected_loadings=pd.read_csv(paths[4]);pcaerrors=[]
    for row in expected_loadings.itertuples():
        if pd.notna(row.z_score):
            match=loadings[loadings.z_score==row.z_score]
            for column in ('pc1_loading','pc2_loading','pc3_loading'):
                # PC2/PC3 eigenvector signs are arbitrary; compare magnitudes for these.
                e=float(getattr(row,column));a=float(match[column].iloc[0]);error=abs(a-e) if column=='pc1_loading' else abs(abs(a)-abs(e))
                pcaerrors.append({'metric':f'{row.z_score}/{column}','expected':e,'reproduced':a,'absolute_error':error,'passed':error<=1e-10})
        if pd.notna(row.z_score_1) and pd.notna(row.z_score_2):
            a=float(loadings.loc[(loadings.z_score_1==row.z_score_1)&(loadings.z_score_2==row.z_score_2),'r'].iloc[0]);e=float(row.r)
            pcaerrors.append({'metric':f'{row.z_score_1}/{row.z_score_2}/r','expected':e,'reproduced':a,'absolute_error':abs(a-e),'passed':abs(a-e)<=1e-10})
    zcomplete=linked.loc[complete,list(ZCOLS)].to_numpy(float)
    svd_score,svd_weights,svd_variance=pca_svd(zcomplete)
    original_weights=unique_pc1_weights(loadings)
    svd_check={'score_max_error':float(np.max(abs(svd_score-computed.loc[complete,'Z_pc1'].to_numpy()))),
               'weight_max_error':float(np.max(abs(svd_weights-original_weights))),
               'variance_error':abs(svd_variance-float(diagnostics.loc[diagnostics.metric=='pc1_explained_variance_ratio','value'].iloc[0]))}
    # Check historical reported descriptives at their four-decimal saved precision.
    saved_desc=pd.read_csv(paths[5]);desc_check=[]
    for row in saved_desc.itertuples():
        sub=linked if row.subgroup=='all' else linked.loc[ls==row.subgroup]
        if row.variable not in sub:continue
        vals=pd.to_numeric(sub[row.variable],errors='coerce').dropna()
        for metric,a in (('n',len(vals)),('mean',vals.mean()),('sd',vals.std(ddof=1)),('median',vals.median()),('min',vals.min()),('max',vals.max())):
            e=float(getattr(row,metric));error=abs(float(a)-e)
            desc_check.append({'metric':f'{row.subgroup}/{row.variable}/{metric}','absolute_error':error,'passed':error<=(0 if metric=='n' else 0.00005+1e-10)})
    # Exact original aggregate validation function, with its write callback disabled.
    pipeline_tree=ast.parse(paths[1].read_text())
    validation_fn=next(n for n in pipeline_tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_zscores')
    validation_namespace={'np':np,'pd':pd,'stats':stats,'log':lambda *args:None,'save_csv':lambda *args:None}
    exec(compile(ast.Module(body=[validation_fn],type_ignores=[]),str(paths[1]),'exec'),validation_namespace)
    validation=validation_namespace['validate_zscores'](linked)
    saved_validation=pd.read_csv(paths[6]);validation_checks=[]
    keys=['analysis','z_score','adjustment_variable','z_score_1','z_score_2']
    current=validation.fillna('').set_index(keys);expected_validation=saved_validation.fillna('').set_index(keys)
    assert not current.index.duplicated().any() and set(current.index)==set(expected_validation.index)
    for key,row in expected_validation.iterrows():
        for metric in ('r','p_value','n'):
            error=abs(float(current.loc[key,metric])-float(row[metric]))
            validation_checks.append({'metric':'/'.join(key)+'/'+metric,'absolute_error':error,'passed':error<=(0 if metric=='n' else .00005+1e-10)})
    # Fixed published model structure refit: no new term selection.
    coeff=pd.read_csv(paths[8]);scales=pd.read_csv(paths[9]);difference_rows=[];refit_rows=[];quality=[]
    x=design(z)
    for group in ('female','male'):
        for j,(lift,raw,zcol) in enumerate(zip(LIFTS,RAW,ZCOLS)):
            force=pd.to_numeric(z[raw],errors='coerce').to_numpy(float)
            valid=(sex==group)&np.isfinite(x).all(1)&np.isfinite(force)&(force>0)
            existing=(sex==group)&np.isfinite(zarray[:,j]);both=valid&existing
            quality.append({'sex':group,'lift':lift,'available_raw':int(valid.sum()),'available_historical_z':int(existing.sum()),
                'nonpositive_raw':int(((sex==group)&np.isfinite(force)&(force<=0)).sum()),
                'raw_invalid_but_z_present':int(((sex==group)&~valid&np.isfinite(zarray[:,j])).sum()),
                'valid_raw_but_z_missing':int((valid&~np.isfinite(zarray[:,j])).sum())})
            candidates={'published_rounded':formula[:,j]}
            c=coeff[(coeff.stratum==group)&(coeff.lift==lift)&(coeff.variant=='reduced')].set_index('term').coefficient
            terms=('intercept','height_cm','age_years','age_squared','log_mass_kg','log_mass_squared')
            eta=x@np.array([c[k] for k in terms])
            scale=scales[(scales.stratum==group)&(scales.lift==lift)&(scales.variant=='reduced')].iloc[0]
            for definition in ('sd_residuals','sigma_residual'):
                predicted=np.full(len(z),np.nan);predicted[valid]=(np.log(force[valid])-eta[valid])/float(scale[definition]);candidates['archived_reduced_'+definition]=predicted
            # Stable OLS via centred/scaled predictor design, then back-transform coefficients.
            xv=x[valid];yv=np.log(force[valid]);mu=xv[:,1:].mean(0);sd=xv[:,1:].std(0,ddof=1)
            standardized=np.column_stack([np.ones(len(xv)),(xv[:,1:]-mu)/sd])
            b,_,rank,_=np.linalg.lstsq(standardized,yv,rcond=None)
            beta=np.r_[b[0]-np.sum(b[1:]*mu/sd),b[1:]/sd]
            residual=yv-xv@beta
            rsd=float(residual.std(ddof=1));sigma=float(np.sqrt(np.sum(residual**2)/(len(residual)-rank)))
            refit=np.full(len(z),np.nan);refit[valid]=residual/rsd;candidates['fixed_published_terms_refit_sd']=refit
            fitted_coeff={k:float(v) for k,v in zip(terms,beta)}
            refit_rows.append({'sex':group,'lift':lift,'n':int(valid.sum()),'rank':int(rank),'coefficients':fitted_coeff,
                              'sd_residuals':rsd,'sigma_residual':sigma,'published_scale':constants[(group,lift)]['rsd'],
                              'r_squared':float(1-np.sum(residual**2)/np.sum((yv-yv.mean())**2))})
            for label,predicted in candidates.items():
                delta=predicted[both]-zarray[both,j]
                difference_rows.append({'sex':group,'lift':lift,'reference':label,'n':int(both.sum()),
                    'mean_difference':float(delta.mean()),'rmse':float(np.sqrt(np.mean(delta**2))),'max_absolute_difference':float(np.max(abs(delta))),
                    'historical_mean':float(zarray[both,j].mean()),'historical_sd':float(zarray[both,j].std(ddof=1))})
    # Compare reconstructed fixed-formula PC1 separately; never substitute it for historical PC1.
    linked_positions=z.index[z.code.isin(t.Probandencode)]
    assert len(linked_positions)==len(linked)
    formula_linked=pd.DataFrame(formula,index=z.code).loc[linked.code].to_numpy()
    formula_complete=np.isfinite(formula_linked).all(1)
    assert np.array_equal(formula_complete,complete.to_numpy())
    formula_pc,formula_weights,formula_variance=pca_svd(formula_linked[formula_complete])
    pc1_reference={'n_complete':int(complete.sum()),'weights':original_weights.tolist(),'variance_fraction':svd_variance,
     'means':zcomplete.mean(0).tolist(),'sds':zcomplete.std(0,ddof=1).tolist(), 'svd_check':svd_check,
     'published_formula_separate_comparison':{'variance_fraction':formula_variance,'weights':formula_weights.tolist(),
     'correlation_with_historical_pc1':float(np.corrcoef(formula_pc,computed.loc[complete,'Z_pc1'])[0,1]),
     'rmse_vs_historical_pc1':float(np.sqrt(np.mean((formula_pc-computed.loc[complete,'Z_pc1'].to_numpy())**2)))}}
    assert before=={str(p):sha(p) for p in paths}
    status={'pc1_diagnostics_passed':all(r['passed'] for r in dcompare),'pc1_weights_and_correlations_passed':all(r['passed'] for r in pcaerrors),
     'historical_descriptives_passed':all(r['passed'] for r in desc_check),
     'historical_z_validation_passed':all(r['passed'] for r in validation_checks),
     'historical_z_reconstruction_passed':all(r['max_absolute_difference']<=1e-10 for r in difference_rows if r['reference']=='archived_reduced_sd_residuals'),
     'svd_check_passed':max(svd_check.values())<1e-10,
     'source_files_unchanged':True,'synthesis_started':False,'person_rows_exported':False,'published':False}
    results={'flow':flow,'input_quality':input_quality,'quality':quality,'pc1':pc1_reference,'pc1_diagnostic_comparison':dcompare,'pc1_loading_comparison':pcaerrors,
             'historical_z_validation_comparison':validation_checks,
             'historical_descriptive_comparison':desc_check,'score_reference_differences':difference_rows,'fixed_published_terms_refits':refit_rows,'checks':status}
    (HERE/'AGGREGATE_RESULTS.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    (HERE/'INPUT_MANIFEST.json').write_text(json.dumps({'files':before,'original_pc1_function_sha256':original_function_sha,
     'scope':'Private local empirical audit; no row-level data copied or exported','runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}},indent=2)+'\n')
    print(json.dumps({'flow':flow,'pc1':pc1_reference,'checks':status},indent=2))

if __name__=='__main__':main()
