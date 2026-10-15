"""Independent aggregation from artificial aggregate records; no study helpers imported."""
import os
from pathlib import Path
import json, hashlib, math, statistics
import numpy as np
from scipy.stats import t

ROOT=Path(os.environ['STRENGTH_STUDY_OUTPUT'])
SOURCE=Path(__file__).resolve().parents[1]/'source'
TOL = 1e-10
MODELS = ('original', 'heavy_tails', 'shifted_mixed')
METHODS = ('depth8_outcomes_first', 'depth8_interleaved')
LEVELS = ('exact', 'mild', 'strong')
KINDS = ('cart', 'copy_positive', 'independent_negative')
PRIMARY = ['link_excess'] + [f'inference_{p}_{l}_5_excess' for p in ('demographics','extended') for l in ('SQ','BP','DL')]
results = json.loads((ROOT/'results.json').read_text())
complete = json.loads((ROOT/'COMPLETE.json').read_text())
limits = results['limits']
counts = dict(utility_rows=0, privacy_conditions=0, utility_flags=0, means_checked=0, interval_nodes=0, control_tests=0)
max_error = 0.
cohort_summaries = {}
joint = {}
def near(a,b):
 global max_error
 delta=abs(a-b);max_error=max(max_error,delta);assert delta <= TOL,(a,b)
def check_utility(rows, summary):
 assert len(rows)==summary['n_expected']
 good=[r for r in rows if r['status']=='ARTIFICIAL_CANDIDATE_SCREEN']
 assert summary['failures']==len(rows)-len(good)
 assert summary['all_passes']==sum(r['all_passed'] for r in rows)
 near(summary['all_pass_fraction'],sum(r['all_passed'] for r in rows)/len(rows))
 for row in rows:
  counts['utility_rows']+=1
  if row['status']=='FAILED_EVALUATION':assert row['all_passed'] is False;continue
  union=set()
  for group, flags in row['checks'].items():
   vals=row['common_audit'] if group=='common' else row['groups'][group]
   assert set(vals)==set(flags)
   for key,value in vals.items():
    assert key in limits and math.isfinite(value)
    assert flags[key] is bool(value <= limits[key]+1e-12)
    union.add(key);counts['utility_flags']+=1
  assert union==set(limits)
  assert row['all_passed']==all(all(v.values()) for v in row['checks'].values())
 for key in limits:
  expected=sum(all(v[key] for v in r['checks'].values() if key in v) for r in good)/len(rows)
  near(summary['criterion_pass_fractions'][key],expected)
 for block in ('all','female','male','common'):
  near(summary['block_pass_fractions'][block],sum(all(r['checks'][block].values()) for r in good)/len(rows))

 for label, saved in summary['mean_losses'].items():
  block,key=label.split('/')
  values=[r['common_audit'][key] if block=='common' else r['groups'][block][key] for r in good]
  assert saved['n_evaluated']==len(values)
  near(saved['mean'],statistics.mean(values));counts['means_checked']+=1
 for key,saved in summary['relationship_diagnostics'].items():
  values=[r['relationship_diagnostics'][key]['absolute_change'] for r in good]
  valid=[x for x in values if x is not None]
  assert saved['n_evaluated']==len(valid) and saved['n_expected']==len(rows)
  if valid:near(saved['mean'],statistics.mean(valid));counts['means_checked']+=1

for model in MODELS:
 cohort_summaries[model]=[]
 joint[model]={method:{level:[] for level in LEVELS} for method in METHODS}
 for c in range(10):
  rawpath=ROOT/'cohorts'/f'{model}_{c}.json'
  raw=json.loads(rawpath.read_text());summary=json.loads(rawpath.with_name(rawpath.stem+'_summary.json').read_text())
  assert hashlib.sha256(rawpath.read_bytes()).hexdigest()==summary['raw_sha256']
  assert raw['limits']==limits and raw['data_status']=='ARTIFICIAL_ONLY'
  assert (raw['n_female'],raw['n_male'])==(197,196)
  assert raw['seeds']['reference']==111000000+c and raw['seeds']['audit']==113000000+c
  assert all(len(rows)==100 for rows in raw['natural'].values())
  for mode,rows in raw['natural'].items():check_utility(rows,summary['natural'][mode])
  assert len(raw['records'])==222
  assert len({(r['method'],r['replica']) for r in raw['records']})==222
  for method in METHODS:
   for kind in KINDS:
    rows=[r for r in raw['records'] if r['method']==method and r['kind']==kind]
    assert len(rows)=={'cart':100,'copy_positive':1,'independent_negative':10}[kind]
    check_utility([r['utility'] for r in rows],summary['methods'][method][kind]['utility'])
    for level in LEVELS:
     counts['privacy_conditions']+=len(rows)
     assert all(r['privacy'][level]['status']=='EVALUATED_NOT_A_PRIVACY_PASS' for r in rows)
     saved=summary['methods'][method][kind]['privacy'][level]
     assert saved['n_evaluated']==saved['n_expected']==len(rows)
     for attack in ('frozen','adapted'):
      for key in saved['means'][attack]:
       values=[r['privacy'][level][attack][key] for r in rows]
       near(saved['means'][attack][key],math.fsum(values)/len(values));counts['means_checked']+=1
       if key.endswith('_excess'):
        prefix=key[:-7]
        for r in rows:near(r['privacy'][level][attack][key],r['privacy'][level][attack][prefix+'_member']-r['privacy'][level][attack][prefix+'_control'])
     if kind=='cart':
      passing=[r for r in rows if r['utility']['all_passed']]
      joint[model][method][level].extend([r['privacy'][level] for r in passing])
     if kind=='copy_positive':
      assert rows[0]['utility']['all_passed']
      if level=='exact':
       near(rows[0]['privacy'][level]['frozen']['link_member'],1.)
       for p in ('demographics','extended'):
        for lift in ('SQ','BP','DL'):near(rows[0]['privacy'][level]['frozen'][f'inference_{p}_{lift}_5_member'],1.)
  cohort_summaries[model].append(summary)

indices=np.random.default_rng(131000000).integers(10,size=(10000,10))
def verify_estimate(node,expected):
 assert node['n_cohorts']==10 and len(node['cohort_means'])==10
 for a,b in zip(node['cohort_means'],expected):near(a,b)
 mu=statistics.mean(expected);sd=statistics.stdev(expected)
 near(node['mean'],mu);near(node['cohort_sd'],sd)
 half=float(t.ppf(.975,9))*sd/math.sqrt(10)
 near(node['t95'][0],mu-half);near(node['t95'][1],mu+half)
 bs=np.quantile(np.array(expected)[indices].mean(1),[.025,.975])
 for a,b in zip(node['bootstrap95'],bs):near(a,float(b))
 counts['interval_nodes']+=1

for model in MODELS:
 cs=cohort_summaries[model];r=results['models'][model]
 for name,node in r['utility'].items():
  source=[c['natural'][name] if name in ('independent','bootstrap') else c['methods'][name]['cart']['utility'] for c in cs]
  assert node['passes']==sum(x['all_passes'] for x in source) and node['intended']==1000
  verify_estimate(node['all_pass'],[x['all_pass_fraction'] for x in source])
  for key in limits:verify_estimate(node['criteria'][key],[x['criterion_pass_fractions'][key] for x in source])
  for block in ('all','female','male','common'):verify_estimate(node['blocks'][block],[x['block_pass_fractions'][block] for x in source])
 for method in METHODS:
  for comparator in ('independent','bootstrap'):
   verify_estimate(r['paired_utility_differences'][method+'_minus_'+comparator],[c['methods'][method]['cart']['utility']['all_pass_fraction']-c['natural'][comparator]['all_pass_fraction'] for c in cs])
 for name,node in r['privacy'].items():
  method,level,kind=name.split('/')
  source=[c['methods'][method][kind]['privacy'][level]['means'] for c in cs]
  for attack in ('frozen','adapted'):
   for key in source[0][attack]:verify_estimate(node[attack][key],[x[attack][key] for x in source])
  for key in PRIMARY:verify_estimate(node['paired_adapted_minus_frozen'][key],[x['adapted'][key]-x['frozen'][key] for x in source])

pvalues=[]
for row in results['control_tests']:
 vals=results['models'][row['model']]['privacy'][row['method']+'/'+row['knowledge']+'/independent_negative'][row['attack']][row['endpoint']]['cohort_means']
 sd=statistics.stdev(vals);mu=statistics.mean(vals)
 p=(1. if mu==0 else 0.) if sd==0 else float(2*t.sf(abs(mu/(sd/math.sqrt(10))),9))
 near(row['p'],p);pvalues.append(p)
order=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);last=0.
for rank,i in enumerate(order):
 last=max(last,min(1.,(len(order)-rank)*pvalues[i]));near(results['control_tests'][i]['holm_p'],last);counts['control_tests']+=1
for model,methods in joint.items():
 for method,levels in methods.items():
  for level,rows in levels.items():
   levels[level]={'n_utility_passing_releases':len(rows),'absolute_and_excess_means':{attack:{key:statistics.mean(r[attack][key] for r in rows) for key in rows[0][attack]} for attack in ('frozen','adapted')} if rows else None,'status':'DESCRIPTIVE_CONDITIONAL_ON_UTILITY_PASS' if rows else 'NOT_ESTIMABLE_NO_UTILITY_PASSES'}
(ROOT/'joint_conditional_results.json').write_text(json.dumps(joint,indent=2)+'\n')
for n,record in json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())['files'].items():assert hashlib.sha256((SOURCE/n).read_bytes()).hexdigest()==record['sha256']
for n,h in complete['source_sha256'].items():assert hashlib.sha256((SOURCE/n).read_bytes()).hexdigest()==h
assert counts['utility_rows']==12660 and counts['privacy_conditions']==19980 and counts['control_tests']==252
out={'status':'PASS','absolute_tolerance':TOL,'maximum_absolute_difference':max_error,'checks':counts,'sources_unchanged':True,'no_person_rows_read':True,'independent_scope':'Independent raw-aggregate means/pass flags, cohort SD/t/bootstrap intervals and Holm arithmetic; not an independent attack implementation.'}
(ROOT/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
