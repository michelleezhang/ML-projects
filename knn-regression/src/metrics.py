import numpy as np

def mean_squared_error(estimates, targets):
    """
    Mean squared error measures the average of the square of the errors (the
    average squared difference between the estimated values and what is
    estimated.

    Refer to the slides, or read more here:
      https://en.wikipedia.org/wiki/Mean_squared_error

    Do not import or use these packages: scipy, sklearn, sys, importlib.

    Args:
        estimates (np.ndarray): the estimated values (should be the same shape as targets)
        targets (np.ndarray): the ground truth values

    Returns:
        MSE (float): the mean squared error across all estimates and targets
    """

    # calculate summation over N of (y_i - h(x_i))^2, divided by N 
    MSE = np.sum((targets - estimates)**2) / targets.shape[0]
    return MSE
