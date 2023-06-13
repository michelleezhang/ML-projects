import numpy as np
import src.random


class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement PolynomialRegression from scratch.
        
        The `degree` argument controls the complexity of the function.  For
        example, degree = 2 would specify a hypothesis space of all functions
        of the form:

            f(x) = ax^2 + bx + c

        You should implement the closed form solution of least squares:
            w = (X^T X)^{-1} X^T y
        
        Do not import or use these packages: scipy, sklearn, sys, importlib.
        Do not use these numpy or internal functions: polynomial, polyfit, polyval, getattr, globals

        Args:
            degree (int): Degree used to fit the data.
        """
        self.degree = degree

        # initialize the weights with the correct dimensions
        self.weights = np.zeros((self.degree+1))

        # self.features = []
    
    def fit(self, features, targets):
        """
        Fit to the given data.

        Hints:
          - Remember to use `self.degree`
          - Remember to include an intercept (a column of all 1s) before you
            compute the least squares solution.
          - If you are getting `numpy.linalg.LinAlgError: Singular matrix`,
            you may want to compute a "pseudoinverse" or add a tiny bit of
            random noise to your input data.

        Args:
            features (np.ndarray): an array of shape [N, 1] containing real-valued inputs.
            targets (np.ndarray): an array of shape [N, 1] containing real-valued targets.
        Returns:
            None (saves model weights to `self.weights`)
        """

        new_features = np.ones((features.size, 1))

        for i in range(1, self.degree+1):
            new_features = np.append(new_features, features**i, axis=1)
            
        self.weights = np.dot(np.linalg.pinv(np.dot(new_features.T, new_features)), np.dot(new_features.T, targets))

        # self.features = np.ones((features.size, 1))

        # for i in range(1, self.degree+1):
        #     self.features = np.append(self.features, features**i, axis=1)
            
        # self.weights = np.dot(np.linalg.pinv(np.dot(self.features.T, self.features)), np.dot(self.features.T, targets))

    def predict(self, features):
        """
        Given features, use the trained model to predict target estimates. Call
        this only after calling fit so that the model has its weights.

        Args:
            features (np.ndarray): array of shape [N, 1] containing real-valued inputs.
        Returns:
            predictions (np.ndarray): array of shape [N, 1] containing real-valued predictions
        """
        assert hasattr(self, "weights"), "Model hasn't been fit!"


        new_features = np.ones((features.size, 1))
        for i in range(1, self.degree+1):
            new_features = np.append(new_features, features**i, axis=1)
        
        predictions = np.dot(new_features, self.weights)
        return predictions

        # predictions = np.matmul(self.features, self.weights)
        # return predictions