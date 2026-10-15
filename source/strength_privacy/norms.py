"""Fixed artificial study kernels; see docs/KERNEL_PROVENANCE.json."""
import numpy as np

LIFTS = ('SQ', 'BP', 'DL')

NORM_STATUS = 'PROVISIONAL_ROUNDED_UNVERIFIED_RSD'

COEFFICIENTS = {'female': {'BP': (-41.41, -0.01937, 0.04317, -0.00073, 21.74, -2.455), 'SQ': (-32.84, -0.02515, 0.03247, -0.00068, 18.26, -2.016), 'DL': (-32.99, -0.01657, 0.05272, -0.00092, 17.86, -2.002)}, 'male': {'BP': (-26.23, -0.0158, 0.00576, -0.00018, 13.62, -1.357), 'SQ': (-29.93, -0.02408, -0.00105, -0.00015, 15.76, -1.554), 'DL': (-33.51, -0.01987, 0.00137, -0.00015, 17.43, -1.781)}}

RSD = {'female': {'BP': 0.267, 'SQ': 0.276, 'DL': 0.269}, 'male': {'BP': 0.193, 'SQ': 0.23, 'DL': 0.219}}

def log_expected(sex, lift, height, age, mass):
    if sex not in COEFFICIENTS or lift not in LIFTS:
        raise ValueError('Unknown sex or lift')
    values = np.asarray([height, age, mass], dtype=float)
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('Finite positive height, age and mass required')
    lm = np.log(mass)
    return float(np.dot(COEFFICIENTS[sex][lift], [1, height, age, age * age, lm, lm * lm]))

def z_score(strength, sex, lift, height, age, mass):
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError('Finite positive 1RM required')
    return float((np.log(strength) - log_expected(sex, lift, height, age, mass)) / RSD[sex][lift])

def make_artificial(seed=42):
    """Invented equations, not calibrated to or sampled from original people."""
    rng = np.random.default_rng(seed)
    sex = np.array(['female'] * 197 + ['male'] * 196)
    rng.shuffle(sex)
    male = (sex == 'male').astype(float)
    age = rng.uniform(18, 60, 393)
    height = np.clip(165 + 13 * male + rng.normal(0, 6, 393), 145, 205)
    mass = np.clip(65 + 15 * male + 0.5 * (height - 170) + rng.normal(0, 7, 393), 45, 125)
    factor = rng.normal(0, 0.8, 393)
    latent = factor[:, None] + rng.normal(0, 0.35, (393, 3))
    strength = np.empty_like(latent)
    scores = np.empty_like(latent)
    for i in range(393):
        for j, lift in enumerate(LIFTS):
            eta = log_expected(sex[i], lift, height[i], age[i], mass[i])
            strength[i, j] = np.exp(eta + RSD[sex[i]][lift] * latent[i, j])
            scores[i, j] = z_score(strength[i, j], sex[i], lift, height[i], age[i], mass[i])
    return {'ids': [f'ARTIFICIAL-{seed}-{i}' for i in range(393)], 'sex': sex.tolist(), 'age': age, 'height': height, 'mass': mass, 'strength': strength, 'z': scores}
