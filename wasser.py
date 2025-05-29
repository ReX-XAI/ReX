from rex_xai.output.database import db_to_pandas
import scipy.stats as stats
import numpy as np

def sliced_wasserstein(X, Y, num_proj):
    '''Takes:
        X: 2d (or nd) histogram
        Y: 2d (or nd) histogram
        num_proj: Number of random projections to compute the mean over
        ---
        returns:
        mean_emd_dist'''
    #% Implementation of the (non-generalized) sliced wasserstein (EMD) for 2d distributions as described here: https://arxiv.org/abs/1902.00434 %#
    # X and Y should be a 2d histogram 
    # Code adapted from stackoverflow user: Dougal - https://stats.stackexchange.com/questions/404775/calculate-earth-movers-distance-for-two-grayscale-images
    dim = X.shape[1]
    ests = []
    
    for _ in range(num_proj):
        
        # sample uniformly from the unit sphere
        dir = np.random.rand(dim)
        dir /= np.linalg.norm(dir)

        # project the data
        X_proj = X @ dir
        Y_proj = Y @ dir
        
        # compute 1d wasserstein
        ests.append(stats.wasserstein_distance(np.arange(dim), np.arange(dim), X_proj, Y_proj))
    return np.mean(ests)


df = db_to_pandas('test.db')
r1 = df.iloc[0]['responsibility']
r2 = df.iloc[1]['responsibility']

print(sliced_wasserstein(r1 / r1.sum(), r2 / r2.sum(), 10000))
print(sliced_wasserstein(r1 / r1.sum(), r1 / r1.sum(), 10000))
