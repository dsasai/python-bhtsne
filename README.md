# Python BHTSNE

Python module for Barnes-Hut implementation of t-SNE (Cython).

This module is based on the excellent work of [Laurens van der Maaten](https://github.com/lvdmaaten/bhtsne).

## Features

- Better results than the Scikit-Learn BH t-SNE implementation: [bhtsne](https://raw.githubusercontent.com/dominiek/python-bhtsne/master/test/plots/iris.png) VS [Scikit-Learn](https://raw.githubusercontent.com/dominiek/python-bhtsne/master/test/plots/iris_sklearn.png)
- Fast (C++/Cython)
- Ability to set random seed
- Ability to set pre-defined plot coordinates (allow for smooth transitions between plots)

## Installation

From pip:

```bash
pip install bhtsne
```

## Examples

### Iris Data Set

Reduce the four dimensional iris data set to two dimensions:

```python
from bhtsne import tsne
from sklearn.datasets import load_iris
iris = load_iris()
Y = tsne(iris.data)
plt.scatter(Y[:, 0], Y[:, 1], c=iris.target)
plt.show()
```

This should result in:

![Iris Plot](https://raw.githubusercontent.com/dominiek/python-bhtsne/master/test/plots/iris.png)


### Transition between two t-SNE results

When adding new data the t-SNE plot can change dramatically (even when settings a random seed). This makes it hard to animate between different plots when data is in motion.

This problem can be partially solved by setting the start coordinates of the first N vectors. In this example we'll create two t-SNE plots, the first one will have part of the iris data set. The second will include the remaining 10 of the iris set:

```python
from bhtsne import tsne
from sklearn.datasets import load_iris
iris = load_iris()
X_a = load_iris().data[:-10]
X_b = load_iris().data
Y_a = tsne(X_a, rand_seed=999)
Y_b = tsne(X_b, seed_positions=Y_a)
plt.scatter(Y_a[:, 0], Y_a[:, 1], c='b')
plt.scatter(Y_b[:-10, 0], Y_b[:-10, 1], c='r')
plt.scatter(Y_b[-10:, 0], Y_b[-10:, 1], c='g')
plt.show()
```

The resulting plot shows our first iteration in blue. Then the second iteration is shown in red and the new nodes that were added are green:

![Iris Plot](https://raw.githubusercontent.com/dominiek/python-bhtsne/master/test/plots/iris_seed_positions.png)

## Development

Build:

```bash
pip install cython
make
```

To run unit tests:

```bash
make test
```

Also creates visual plots in the `test/plots` folder.

## Todo

- Allow more sophisticated control of updates to the t-SNE (streaming/online t-SNE)
