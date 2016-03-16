# Python BHTSNE

Python module for Barnes-Hut implementation of t-SNE (Cython).

This module is based on the excellent work of [Laurens van der Maaten](https://github.com/lvdmaaten/bhtsne).

## Features

- [Better results](https://raw.githubusercontent.com/dominiek/python-bhtsne/master/test/plots/iris_sklearn.png) than the Scikit-Learn BH t-SNE implementation
- Fast (C++/Cython)
- Ability to set random seed
- Ability to set pre-defined plot coordinates (allow for smooth transitions between plots)

## Installation

From pip:

```bash
pip install bhtsne
```

Custom build:

```bash
make
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

## Tests

To run unit tests:

```bash
make test
```

Also creates visual plots in the `test/plots` folder.

## Todo

- Allow more sophisticated control of updates to the t-SNE (streaming/online t-SNE)
