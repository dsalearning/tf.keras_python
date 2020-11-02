import numpy as np
from scipy.stats import zscore

x = np.array([255, 128, 45, 0])

z_score = zscore(x)
print(z_score)

print(zscore([[1, 2, 3],
              [6, 7, 8]], axis=1))