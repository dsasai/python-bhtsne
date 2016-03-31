
import numpy as np
from bhtsne_wrapper import BHTSNE

class InvalidSeedPositionError(Exception):
    pass

class TSNE(object):

    def __init__(self, data, dimensions=2, rand_seed=-1, seed_positions=np.array([])):
        skip_random_init = False
        if len(seed_positions) > 0:
            skip_random_init = True
            if seed_positions.shape[0] != data.shape[0]:
                raise InvalidSeedPositionError("Seed positions needs to be same number of rows as input matrix")

        self._reduction = np.zeros((data.shape[0], dimensions), dtype=np.float64)
        self._tsne = BHTSNE(data, data.shape[0], data.shape[1], self._reduction, dimensions, rand_seed, seed_positions, skip_random_init)

    def fit(self, perplexity=30.0, theta=0.5):
        self._tsne.fit(perplexity=perplexity, theta=theta)
        return self._reduction
