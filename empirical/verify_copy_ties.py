"""Aggregate diagnostic for exact-match ties; no person-row export."""
import numpy as np
import run as r
import empirical as e

def main():
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
