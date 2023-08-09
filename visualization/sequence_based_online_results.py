from sklearn.preprocessing import normalize
import numpy as np

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])
print(X)
X = normalize(X, axis=0, norm='l1')
print(X)