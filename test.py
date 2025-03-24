import torch
import numpy as np
total_samples = sum([2201, 1263, 843, 429, 428, 452, 423, 422, 854, 431])
weights = np.array([total_samples / c for c in [2201, 1263, 843, 429, 428, 452, 423, 422, 854, 431]])
weights=weights/np.sum(weights)
print(weights)
