"""Fixed artificial study kernels; see docs/KERNEL_PROVENANCE.json."""
import numpy as np
from . import norms as s, scores as q

def features(data):
    male = (np.array(data['sex']) == 'male').astype(float)
    a = (data['age'] - 39) / 21
    h = (data['height'] - (165 + 13 * male)) / 6
    m = (data['mass'] - (65 + 15 * male + 0.5 * (data['height'] - 170))) / 7
    return np.column_stack([a * np.sqrt(3), h, m, (a * a - 1 / 3) * np.sqrt(45 / 4), h * m / np.sqrt(1 + a * a)])

def signal(data, scenario):
    x = features(data)
    if scenario == 'linear':
        return 0.5 * x[:, :3]
    if scenario == 'nonlinear':
        return 0.5 * np.column_stack([x[:, 3], x[:, 4], (x[:, 3] + x[:, 4]) / np.sqrt(2)])
    raise ValueError('Unknown scenario')

def make(seed, scenario):
    data = s.make_artificial(seed)
    q.encode_scores(data, data['z'] + signal(data, scenario))
    return data
