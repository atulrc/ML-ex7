# Scientific and vector computation for python
import numpy as np

def proData(X, U, K):
    """
    Computes the reduced data representation when projecting only
    on to the top K eigenvectors.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n). The dataset is assumed to be
        normalized.

    U : array_like
        The computed eigenvectors using PCA. This is a matrix of
        shape (n x n). Each column in the matrix represents a single
        eigenvector (or a single principal component).

    K : int
        Number of dimensions to project onto. Must be smaller than n.

    Returns
    -------
    Z : array_like
        The projects of the dataset onto the top K eigenvectors.
        This will be a matrix of shape (m x k).

    Instructions
    ------------
    Compute the projection of the data using only the top K
    eigenvectors in U (first K columns).
    For the i-th example X[i,:], the projection on to the k-th
    eigenvector is given as follows:

        x = X[i, :]
        projection_k = np.dot(x,  U[:, k])

    """
    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    # ====================== YOUR CODE HERE ======================

    Z = np.dot(X,  U[:, :K])

    # =============================================================
    return Z
