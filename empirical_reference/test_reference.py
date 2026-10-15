import unittest
import numpy as np
import pandas as pd
from reference_utils import join_keys, published_scores, pca_svd, unique_pc1_weights

class ReferenceTests(unittest.TestCase):
    def test_repeated_consistent_aggregate_weights_are_not_extra_lifts(self):
        t=pd.DataFrame({'z_score':['Z_squat','Z_squat','Z_benchpress','Z_deadlift'],'pc1_loading':[.6,.6,.5,.4]})
        self.assertEqual(unique_pc1_weights(t).tolist(),[.6,.5,.4])
        t.loc[1,'pc1_loading']=.7
        with self.assertRaises(ValueError):unique_pc1_weights(t)
    def test_duplicate_and_null_keys_rejected(self):
        for codes in ([1,1],[1,None]):
            with self.assertRaises(ValueError): join_keys(pd.DataFrame({'code':codes}),pd.DataFrame({'Probandencode':[1,2]}))
    def test_join_is_exact_not_normalized(self):
        x=join_keys(pd.DataFrame({'code':['A','B']}),pd.DataFrame({'Probandencode':['A',' B']}))
        self.assertEqual(len(x),1)
    def test_zero_strength_is_missing_not_log_zero(self):
        d=pd.DataFrame({'sex':['weiblich']*3,'age':[25]*3,'bodyheight':[165]*3,'bodymass':[65]*3,'sq1rm':[60,0,-1],'bp1rm':[40]*3,'dl1rm':[80]*3})
        constants={('female',l):{'coefficients':[0]*6,'rsd':1.} for l in ('SQ','BP','DL')}
        z=published_scores(d,constants)
        self.assertAlmostEqual(z[0,0],np.log(60));self.assertTrue(np.isnan(z[1:,0]).all())
    def test_pca_svd_matches_independent_eigendecomposition(self):
        z=np.random.default_rng(4).normal(size=(40,3));z[:,1]+=z[:,0]
        score,w,v=pca_svd(z)
        standardized=(z-z.mean(0))/z.std(0,ddof=1)
        values,vectors=np.linalg.eigh(np.corrcoef(standardized,rowvar=False));weights=vectors[:,-1]
        if weights.sum()<0:weights=-weights
        self.assertTrue(np.allclose(w,weights,atol=1e-12));self.assertAlmostEqual(v,values[-1]/3)
        self.assertAlmostEqual(score.std(ddof=1),1.)

if __name__=='__main__':unittest.main()
