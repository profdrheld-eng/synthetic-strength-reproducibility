import unittest
import tempfile
from artificial_parameters import write_parameters
from empirical import load_reference
_parameters=None
def setUpModule():
    global _parameters
    _parameters=tempfile.TemporaryDirectory()
    write_parameters(_parameters.name)
    load_reference(_parameters.name)
def tearDownModule():
    _parameters.cleanup()

import numpy as np
from empirical import encode,validate,assess,make_fixture,subset,bridge_scores,attacks,balanced_members,split_indices,standardized,cluster_intervals,WEIGHTS,privacy_vectors,METHODS,synthesize

class EmpiricalTests(unittest.TestCase):
    def test_historical_roundtrip_and_copy_utility(self):
        a=make_fixture(1,80)
        validate(a)
        self.assertTrue(assess(a,a,make_fixture(2,30)['z'])['all_passed'])
    def test_link_individual_scores_equal_frozen_kernel(self):
        d=make_fixture(3,30);x=attacks.matrix(d);scale=x.std(0,ddof=1)
        vector=bridge_scores(x[:,:3],x[:,3:],x,scale[:3],scale[3:],1,9)
        expected=attacks.bridge(x[:,:3],x[:,3:],x,scale[:3],scale[3:],1,9)
        self.assertAlmostEqual(vector.mean(),expected)
        self.assertTrue(np.allclose(vector,1.))
    def test_equal_sex_candidates_and_disjoint_folds(self):
        a=make_fixture(4,90);folds=split_indices(a,3)
        self.assertEqual(len(set(np.concatenate(folds))),len(a['ids']))
        h=folds[0];t=np.setdiff1d(np.arange(len(a['ids'])),h)
        m=balanced_members(a,t,h,5)
        self.assertFalse(set(m)&set(h))
        for sex in ('female','male'):
            self.assertEqual(sum(a['sex'][i]==sex for i in m),sum(a['sex'][i]==sex for i in h))
    def test_identifier_overlap_is_rejected(self):
        a=make_fixture(5,30)
        with self.assertRaises(ValueError):attacks.check_disjoint(a,a)
    def test_invalid_raw_score_combination_rejected(self):
        a=make_fixture(6,30);a['z'][0,0]+=.5
        with self.assertRaises(ValueError):validate(a)
    def test_sex_standardization_known_values(self):
        d=make_fixture(7,30);v=np.zeros((2,2,7,30))
        v[:,:,:,np.array(d['sex'])=='female']=1
        self.assertTrue(np.allclose(standardized(v,d,d),WEIGHTS['female']))
    def test_coupled_cluster_interval_known_constant(self):
        d=make_fixture(8,30);den=np.ones((2,30));num=np.zeros((2,2,7,30));num[:,0]=.3;num[:,1]=.1
        self.assertTrue(np.allclose(cluster_intervals(num,den,d['sex']),.2))
    def test_private_copy_attacks_dynamic_cohort(self):
        a=make_fixture(9,90);h=split_indices(a)[0];t=np.setdiff1d(np.arange(90),h);m=balanced_members(a,t,h,10)
        v,s=privacy_vectors(subset(a,m),subset(a,h),subset(a,t),METHODS[0],'exact',11)
        self.assertEqual(v.shape,(2,2,7,len(h)))
        self.assertTrue(np.all(v[0,0]==1))
        self.assertTrue(np.isfinite(standardized(v,subset(a,m),subset(a,h))).all())
    def test_synthesis_historical_reference(self):
        d=make_fixture(12,80)
        for method in METHODS:
            s=synthesize(d,method,13);validate(s)
            self.assertEqual(s['sex'],d['sex'])
            self.assertFalse(set(s['ids'])&set(d['ids']))

if __name__=='__main__':unittest.main()
