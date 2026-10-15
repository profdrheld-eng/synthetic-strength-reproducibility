"""Fixed artificial study kernels; see docs/KERNEL_PROVENANCE.json."""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from . import norms as s, scores as q
VARIANTS = {"depth8_outcomes_first": {"method":"cart","depth":8,"leaf":5,"order":(3,4,5,0,1,2)}, "depth8_interleaved": {"method":"cart","depth":8,"leaf":5,"order":(0,3,1,4,2,5)}}

def eta_matrix(data):
    eta = np.empty((len(data['ids']), 3))
    lm = np.log(data['mass'])
    features = np.column_stack([np.ones(len(lm)), data['height'], data['age'], data['age'] ** 2, lm, lm ** 2])
    for sex in ('female', 'male'):
        mask = np.array(data['sex']) == sex
        for j, lift in enumerate(s.LIFTS):
            eta[mask, j] = features[mask] @ np.asarray(s.COEFFICIENTS[sex][lift])
    return eta

def sample_matrix(x, sex, seed, config):
    rng = np.random.default_rng(seed)
    out = np.empty_like(x)
    for group in ('female', 'male'):
        mask = sex == group
        a = x[mask]
        count = len(a)
        b = np.empty_like(a)
        b[:, 0] = rng.choice(a[:, 0], size=count)
        for j in range(1, a.shape[1]):
            tree = DecisionTreeRegressor(max_depth=config['depth'], min_samples_leaf=config['leaf'], random_state=seed + j)
            tree.fit(a[:, :j], a[:, j])
            leaves = tree.apply(a[:, :j])
            sampled = tree.apply(b[:, :j])
            for leaf in np.unique(sampled):
                pick = sampled == leaf
                b[pick, j] = rng.choice(a[leaves == leaf, j], size=int(pick.sum()))
        out[mask] = b
    return out

def generate(reference, name, seed):
    if name not in VARIANTS:
        raise ValueError('Unknown fixed grid variant')
    config = VARIANTS[name]
    order = np.array(config['order'])
    x = np.column_stack([reference['age'], reference['height'], reference['mass'], reference['z']])
    sampled = sample_matrix(x[:, order], np.array(reference['sex']), seed, config)
    out = sampled[:, np.argsort(order)]
    data = {'ids': [f'ARTIFICIAL-GRID-{name}-{seed}-{i}' for i in range(len(out))], 'sex': reference['sex'].copy(), 'age': out[:, 0], 'height': out[:, 1], 'mass': out[:, 2]}
    rsd = np.array([[s.RSD[sex][lift] for lift in s.LIFTS] for sex in data['sex']])
    data['strength'] = np.exp(eta_matrix(data) + rsd * out[:, 3:])
    q.recalculate(data)
    return data
