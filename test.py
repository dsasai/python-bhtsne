
from bhtsne import tsne
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
X_2d = tsne(X, rand_seed=999)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.show()
