import numpy as np


class Regularizer:
    """
    Regularization for FullyConnected layers
    """
    def __init__(self, alpha=0.01, penalty='l2'):
        """
        penalty: type of distance measure, either "l1" or "l2"
        alpha: weight parameter for the regularization gradient

        You will not need to edit this function.
        """
        self.alpha = alpha
        self.penalty = penalty

    def grad(self, weights):
        if self.penalty == "l1":
            return self.l1_grad(weights)
        elif self.penalty == "l2":
            return self.l2_grad(weights)

    def l1_grad(self, weights):
        """
        Compute the *gradient* of L1 regularization
            with respect to the given weights.

        L1 regularization is just the absolute value, so the partial gradient
            for a given weight is either 1, -1, or 0 depending on whether the
            weight is greater than, less than, or equal to 0.

        Note: weights[0, :] contains the intercept, and you should not apply
            regularization to those parameters.
        """

        l1_gradient = np.zeros((weights.shape[0], weights.shape[1]))

        for i in range(1, weights.shape[0]):
            for j in range(weights.shape[1]):
                if weights[i, j] > 0:
                    l1_gradient[i, j] = 1 
                elif weights[i, j] < 0:
                    l1_gradient[i, j] = -1
                else:
                    l1_gradient[i, j] = 0
        
        return l1_gradient * self.alpha
    

    def l2_grad(self, weights):
        """
        Compute the *gradient* of L2 regularization
            with respect to the given weights.

        L2 regularization is the squared value, so the partial gradient
            for a given weight should be proportional to that weight

        Note: weights[0, :] contains the intercept, and you should not apply
            regularization to those parameters.
        """

        l2_gradient = np.zeros((weights.shape[0], weights.shape[1]))

        for i in range(1, weights.shape[0]):
            for j in range(weights.shape[1]):
                l2_gradient[i, j] = 2 * weights[i, j]
        
        return l2_gradient * self.alpha
