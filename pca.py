import numpy as np


class PCA:
    def __init__(self, n_components):
        """
        PCA class constructor.

        Parameters:
        - n_components: int, the number of principal components to retain
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit the model with X by computing the mean, the principal components, and transforming the data.

        Parameters:
        - X: ndarray, shape (n_samples, n_features), the input data
        """
        # Standardize the range of continuous initial variables
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Compute the covariance matrix
        cov_matrix = np.cov(X.T)

        # Compute the eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort the eigenvectors by decreasing eigenvalues
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Store the first n_components eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        """
        Transform X onto the new space defined by the principal components.

        Parameters:
        - X: ndarray, shape (n_samples, n_features), the input data

        Returns:
        - X_transformed: ndarray, shape (n_samples, n_components), the transformed data
        """
        X = X - self.mean
        X_transformed = np.dot(X, self.components.T)
        return X_transformed
