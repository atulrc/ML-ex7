# Scientific and vector computation for python
import numpy as np

def recData(Z, U, K):
    """
    Recovers an approximation of the original data when using the
    projected data.

    Parameters
    ----------
    Z : array_like
        The reduced data after applying PCA. This is a matrix
        of shape (m x K).

    U : array_like
        The eigenvectors (principal components) computed by PCA.
        This is a matrix of shape (n x n) where each column represents
        a single eigenvector.

    K : int
        The number of principal components retained
        (should be less than n).

    Returns
    -------
    X_rec : array_like
        The recovered data after transformation back to the original
        dataset space. This is a matrix of shape (m x n), where m is
        the number of examples and n is the dimensions (number of
        features) of original datatset.

    Instructions
    ------------
    Compute the approximation of the data by projecting back
    onto the original space using the top K eigenvectors in U.
    For the i-th example Z[i,:], the (approximate)
    recovered data for dimension j is given as follows:

        v = Z[i, :]
        recovered_j = np.dot(v, U[j, :K])

    Notice that U[j, :K] is a vector of size K.
    """
    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    # ====================== YOUR CODE HERE ======================

    X_rec = Z.dot(U[:, :K].T)

    # =============================================================
    return X_rec
