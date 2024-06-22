import numpy as np


def random(min: float = 0, max: float = 1):
    mu = (max + min) / 2  # Mean of the distribution
    sigma = (
        max - min
    ) / 6  # Standard deviation, chosen so that ~99.7% fall within [min_val, max_val]
    sample = np.random.normal(mu, sigma)
    return np.clip(sample, min, max)
