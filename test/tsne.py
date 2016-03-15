
import os
import unittest
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from bhtsne import tsne
from sklearn.datasets import load_iris

PLOTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/plots'

def mean_shift(X):
    bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels_unique = np.unique(ms.labels_)
    return len(labels_unique), ms.cluster_centers_

class TestTsne(unittest.TestCase):

    def test_iris(self):
        iris = load_iris()
        X = iris.data
        self.assertEqual(mean_shift(X)[0], 2)
        X_2d = tsne(X)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=iris.target)

        num_clusters, cluster_centers = mean_shift(X_2d)
        self.assertTrue(num_clusters > 1)
        self.assertTrue(num_clusters < 4)
        for k in range(num_clusters):
            cluster_center = cluster_centers[k]
            plt.plot(cluster_center[0], cluster_center[1], 'x', markerfacecolor='r',
                     markeredgecolor='r', markersize=16)

        plt.savefig(PLOTS_DIR + '/iris.png')
        if os.environ.get('SHOW_PLOTS', None) != None:
            plt.show()
        plt.close()

    def test_set_rand_seed(self):
        iris = load_iris()
        X = iris.data
        X_2d_a = tsne(X, rand_seed=999)
        X_2d_b = tsne(X, rand_seed=999)

        self.assertEqual(round(X_2d_a[0][0]), round(X_2d_b[0][0]))
        self.assertEqual(round(X_2d_a[0][1]), round(X_2d_b[0][1]))

        plt.scatter(X_2d_a[:, 0], X_2d_a[:, 1], c='b')
        plt.scatter(X_2d_b[:, 0], X_2d_b[:, 1], c='r')
        plt.savefig(PLOTS_DIR + '/iris_set_rand_seed.png')
        if os.environ.get('SHOW_PLOTS', None) != None:
            plt.show()
        plt.close()

    def test_without_seed_positions(self):
        iris = load_iris()
        X_a = load_iris().data[:-10]
        X_b = load_iris().data
        X_2d_a = tsne(X_a, rand_seed=999)
        X_2d_b = tsne(X_b, rand_seed=999)

        plt.scatter(X_2d_a[:, 0], X_2d_a[:, 1], c='b')
        plt.scatter(X_2d_b[:-10, 0], X_2d_b[:-10, 1], c='r')
        plt.scatter(X_2d_b[-10:, 0], X_2d_b[-10:, 1], c='g')

        plt.savefig(PLOTS_DIR + '/iris_without_seed_positions.png')
        if os.environ.get('SHOW_PLOTS', None) != None:
            plt.show()
        plt.close()

    def test_seed_positions(self):
        iris = load_iris()
        X_a = load_iris().data[:-10]
        X_b = load_iris().data
        X_2d_a = tsne(X_a, rand_seed=999)
        X_2d_b = tsne(X_b, seed_positions=X_2d_a)

        self.assertEqual(round(X_2d_a[0][0] / 10), round(X_2d_b[0][0] / 10))
        self.assertEqual(round(X_2d_a[0][1] / 10), round(X_2d_b[0][1] / 10))

        plt.scatter(X_2d_a[:, 0], X_2d_a[:, 1], c='b')
        plt.scatter(X_2d_b[:-10, 0], X_2d_b[:-10, 1], c='r')
        plt.scatter(X_2d_b[-10:, 0], X_2d_b[-10:, 1], c='g')

        plt.savefig(PLOTS_DIR + '/iris_seed_positions.png')
        if os.environ.get('SHOW_PLOTS', None) != None:
            plt.show()
        plt.close()
