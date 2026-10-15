"""Additional independent numeric verification; artificial arrays only."""
import json,os,sys
from pathlib import Path
if sys.flags.optimize:raise SystemExit('Run without -O; validation assertions are required')
import numpy as np
import empirical as e
rng=np.random.default_rng(170000001);sex=['female']*11+['male']*13
num=rng.random((2,2,7,24));den=rng.integers(1,9,size=(2,24)).astype(float)
actual=e.cluster_intervals(num,den,sex)
rng=np.random.default_rng(148000000);expected=np.zeros((2000,2,7))
for group,weight in e.WEIGHTS.items():
    idx=np.flatnonzero(np.array(sex)==group);draw=idx[rng.integers(len(idx),size=(2000,len(idx)))]
    for b,indices in enumerate(draw):
        for mode in range(2):
            for j in range(7):
                member=sum(num[mode,0,j,i] for i in indices)/sum(den[0,i] for i in indices)
                control=sum(num[mode,1,j,i] for i in indices)/sum(den[1,i] for i in indices)
                expected[b,mode,j]+=weight*(member-control)
expected=np.quantile(expected,[.025,.975],axis=0).transpose(1,2,0)
error=float(np.max(abs(actual-expected)));assert error<1e-12
target=Path(os.environ['STRENGTH_EMPIRICAL_OUTPUT'])/'ADDITIONAL_VERIFICATION.json'
if e.ROOT.parent.resolve() in target.resolve().parents:raise SystemExit('Choose an output directory outside the repository')
with target.open('x') as f:f.write(json.dumps({'artificial_only':True,'person_cluster_interval_against_explicit_scalar_sums':True,'max_absolute_error':error},indent=2)+'\n')
print('PASS: cluster interval against explicit scalar sums')
