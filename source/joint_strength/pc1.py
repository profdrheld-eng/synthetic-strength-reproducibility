"""Historical PC1 rule, applied only to internally invented profiles here."""
from dataclasses import dataclass

import numpy as np


def matrix(z):
    x = np.asarray(z, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or np.isinf(x).any():
        raise ValueError('Three SQ/BP/DL columns without infinity required')
    return x


@dataclass(frozen=True)
class PC1:
    means: tuple
    sds: tuple
    weights: tuple
    raw_mean: float
    raw_sd: float
    variance_fraction: float
    n_complete: int

    def transform(self, z):
        x = matrix(z)
        complete = np.isfinite(x).all(axis=1)
        result = np.full(len(x), np.nan)
        result[complete] = (((x[complete] - self.means) / self.sds) @ self.weights
                            - self.raw_mean) / self.raw_sd
        return result


def fit_pc1(z):
    x = matrix(z)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 3:
        raise ValueError('At least three complete profiles required')
    means, sds = x.mean(0), x.std(0, ddof=1)
    if np.any(sds <= 0):
        raise ValueError('Constant lift score')
    u = (x - means) / sds
    values, vectors = np.linalg.eigh(np.corrcoef(u, rowvar=False))
    if np.isclose(values[-1], values[-2], rtol=1e-10, atol=1e-12):
        raise ValueError('PC1 is not uniquely identified')
    weights = vectors[:, -1]
    if weights.sum() < 0:
        weights = -weights
    raw = u @ weights
    raw_sd = float(raw.std(ddof=1))
    if raw_sd <= 0 or not np.isfinite(raw_sd):
        raise ValueError('Degenerate PC1')
    return PC1(tuple(means), tuple(sds), tuple(weights), float(raw.mean()), raw_sd,
               float(values[-1] / values.sum()), len(x))
