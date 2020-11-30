import numpy as np


def normal_eqn(X, y):
    """
    normal_eqn Computes the closed-form solution to linear regression
        normal_eqn(x, y) computes the closed-form solution to linear
        regression using the normal equations.
    """

    theta = np.zeros(X.shape[1])

    """
    ====================== YOUR CODE HERE ======================
    Instructions: Complete the code to compute the closed form solution
                  to linear regression and put the result in theta.
    """

    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    # =================================================================
    return theta
