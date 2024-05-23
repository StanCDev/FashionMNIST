import numpy as np

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        X = training_data
        d = self.d
        # Compute the mean of data
        mean = np.mean(X,axis=0)
        self.mean = mean
        # Center the data with the mean
        X_tilde = X - mean
        # Create the covariance matrix
        C = X_tilde.T @ X_tilde
        # Compute the eigenvectors and eigenvalues. Hint: look into np.linalg.eigh()
        eigvals, eigvecs = np.linalg.eigh(C)
        # Choose the top d eigenvalues and corresponding eigenvectors. 
        index = np.argsort(-eigvals)
        eigvals = eigvals[index]
        eigvecs = eigvecs[:,index]

        W = eigvecs[:, :d]
        self.W = W
        eg = eigvals[:d]

        # project the data using W
        # Y = X_tilde @ W
        
        # Compute the explained variance
        exvar = 100 * np.sum(eg) / np.sum(eigvals)
        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        # project the data using W
        data_reduced = (data - self.mean) @ self.W
        return data_reduced
        

