"""Aggregate diagnostic for exact-match ties; no person-row export."""
import numpy as np
import run as r
import empirical as e
import os,sys

def main():
    if sys.flags.optimize:raise SystemExit('Run without -O; validation assertions are required')
    if not os.environ.get('STRENGTH_EMPIRICAL_OUTPUT') or not r.HERE.is_dir():
        raise SystemExit('Existing external empirical output directory required')
    repo=r.SCRIPT_ROOT.parent.resolve()
    if r.HERE.resolve()==repo or repo in r.HERE.resolve().parents:
        raise SystemExit('Choose an output directory outside the repository')
    r.validate_reference_inputs()
    e.load_reference(os.environ['STRENGTH_PARAMETERS'])
    d=r.load();checks=[]
    for fold,h in enumerate(e.split_indices(d)):
        t=np.setdiff1d(np.arange(374),h);m=e.balanced_members(d,t,h,143000000+1000*fold+100)
        copy=e.subset(d,t);member=e.subset(d,m);record={'fold':fold}
        for label,cols in [('body',slice(0,3)),('lifts',slice(3,6))]:
            tied=0
            for sex in ('female','male'):
                x=e.attacks.matrix(copy)[np.array(copy['sex'])==sex,cols]
                y=e.attacks.matrix(member)[np.array(member['sex'])==sex,cols]
                counts=np.all(y[:,None,:]==x[None,:,:],axis=2).sum(1)
                tied+=int(np.sum(counts>1))
            record['member_targets_with_multiple_exact_'+label+'_matches']=tied
        checks.append(record)
    r.save('COPY_TIE_DIAGNOSTIC.json',{'checks':checks,'definition':'Same-sex exact matches on age/height/mass or three lift values in copied release; aggregate verification only'})
if __name__=='__main__':main()
