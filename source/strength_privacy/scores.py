"""Fixed artificial study kernels; see docs/KERNEL_PROVENANCE.json."""
import numpy as np
from . import norms as s

LIMITS = {'z_mean_max': 0.1, 'z_sd_ratio_max': 0.1, 'z_tail_max': 0.2, 'correlation_max': 0.05, 'frozen_pc1_mean': 0.1, 'frozen_pc1_sd_ratio': 0.1, 'frozen_pc1_tail': 0.2, 'missing_fraction_change': 0.05, 'pc1_rmse': 0.15, 'pc1_bias': 0.1, 'pc1_correlation_loss': 0.01, 'variance_fraction_change': 0.05, 'weight_max': 0.05}

def recalculate(data):
    z = np.full((len(data['ids']), 3), np.nan)
    for i, sex in enumerate(data['sex']):
        for j, lift in enumerate(s.LIFTS):
            y = data['strength'][i, j]
            if np.isnan(y):
                continue
            z[i, j] = s.z_score(y, sex, lift, data['height'][i], data['age'][i], data['mass'][i])
    data['z'] = z

def encode_scores(data, z):
    for i, sex in enumerate(data['sex']):
        for j, lift in enumerate(s.LIFTS):
            eta = s.log_expected(sex, lift, data['height'][i], data['age'][i], data['mass'][i])
            data['strength'][i, j] = np.exp(eta + s.RSD[sex][lift] * z[i, j])
    recalculate(data)
