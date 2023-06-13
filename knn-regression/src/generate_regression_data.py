import numpy as np
import src.random


def generate_random_numbers(degree, N, amount_of_noise):
    """
    Args:
        degree (int): degree of Polynomial that relates the output x and y
        N (int): number of points to generate
        amount_of_noise (float): amount of random noise to add to the relationship 
            between x and y.
    
    Returns:     
        Use src.random to generate `explanatory variable x`: an array of shape (N, 1) that contains 
        floats chosen uniformly at random between -1 and 1.
        Use src.random to generate `coefficient value coefs`: an array of shape (degree+1, ) that contains 
        floats chosen uniformly at random between -10 and 10.
        Use src.random to generate `noise variable n`: an array of shape (N, 1) that contains 
        floats chosen in the normal distribution. The mean is 0 and the standard deviation is `amount_of_noise`.

    Note that noise should have std `amount_of_noise`
        which we'll later multiply by `np.std(y)`
    """

    # generate explanatory variable x
    x = src.random.uniform(-1, 1, (N, 1))

    # generate coefficient value coefs
    coeff_val_coeffs = src.random.uniform(-10, 10, (degree+1))

    # generate noise variable n
    n = src.random.normal(0, amount_of_noise, (N, 1))

    return x, coeff_val_coeffs, n


def generate_regression_data(degree, N, amount_of_noise=1.0):
    """

    1. Call `generate_random_numbers` to generate the x values, the
       coefficients of our Polynomial, and the noise.

    2. Use the coefficients to construct a Polynomial function f()
       with the given coefficients.
       If coefficients is array([1, -2, 3]), f(x) = 1 - 2 x + 3 x^2

    3. Compute y0 = f(x) as the output of the regression *without noise*

    4. Create our noisy data `y` as `y0 + noise * np.std(y0)`

    Do not import or use these packages: scipy, sklearn, sys, importlib.
    Do not use these numpy or internal functions: polynomial, polyfit, polyval, getattr, globals

    Args:
        degree (int): degree of Polynomial that relates the output x and y
        N (int): number of points to generate
        amount_of_noise (float): scale of random noise to add to the relationship 
            between x and y.
    Returns:
        x (np.ndarray): explanatory variable of size (N, 1), ranges between -1 and 1.
        y (np.ndarray): response variable of size (N, 1), which responds to x as a
                        Polynomial of degree 'degree'.

    """

    # generate x values, Polynomial coefficients, and noise
    x, coeffs, noise = generate_random_numbers(degree, N, amount_of_noise)

    # helper function to calculate Polynomial value
    def poly(x):
        result = 0
        for i in range(degree+1):
            result += coeffs[i] * x**i
        return result
    
    # create y0, the output of the regression (Polynomial) without noise
    y0 = np.zeros((N, 1))

    for i in range(N):
        y0[i] = poly(x[i])
    
    # create y, the output with noise
    y = y0 + noise * np.std(y0)

    return x, y
