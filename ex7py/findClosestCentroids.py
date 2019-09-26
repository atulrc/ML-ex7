# Scientific and vector computation for python
import numpy as np

def getNearCentroids(X, centroids):
    """
    Computes the centroid memberships for every example.

    Parameters
    ----------
    X : array_like
        The dataset of size (m, n) where each row is a single example.
        That is, we have m examples each of n dimensions.

    centroids : array_like
        The k-means centroids of size (K, n). K is the number
        of clusters, and n is the the data dimension.

    Returns
    -------
    idx : array_like
        A vector of size (m, ) which holds the centroids assignment for each
        example (row) in the dataset X.

    Instructions
    ------------
    Go over every example, find its closest centroid, and store
    the index inside `idx` at the appropriate location.
    Concretely, idx[i] should contain the index of the centroid
    closest to example i. Hence, it should be a value in the
    range 0..K-1

    Note
    ----
    You can use a for-loop over the examples to compute this.
    """
    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros(X.shape[0], dtype=int)

    # ====================== YOUR CODE HERE ======================

    for i in np.arange(idx.size):

        J = np.sqrt(np.sum(np.square(X[i] - centroids), axis = 1))

        idx[i] = np.argmin(J)

    # =============================================================
    return idx
