# Scientific and vector computation for python
import numpy as np

def initCentroids(X, K):
    """
    This function initializes K centroids that are to be used in K-means on the dataset x.

    Parameters
    ----------
    X : array_like
        The dataset of size (m x n).

    K : int
        The number of clusters.

    Returns
    -------
    centroids : array_like
        Centroids of the clusters. This is a matrix of size (K x n).

    Instructions
    ------------
    You should set centroids to randomly chosen examples from the dataset X.
    """
    m, n = X.shape

    # You should return this values correctly
    centroids = np.zeros((K, n))

    # ====================== YOUR CODE HERE ======================

    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]

    # =============================================================
    return centroids
