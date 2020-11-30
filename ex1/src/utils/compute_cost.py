import numpy as np


def compute_cost(X, y, theta):
    """
    compute_cost Compute cost for linear regression
        compute_cost(x, y, theta) computes the cost of using theta as the
        parameter for linear regression to fit the data points in x and y
    """

    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0

    """
    ====================== YOUR CODE HERE ======================
    Instructions: Compute the cost of a particular choice of theta
                  You should set j to the cost.
    """

    h = np.dot(X, theta)

    J = (1 / (2 * m)) * np.sum(np.square(np.dot(X, theta) - y))

    # ============================================================

    return J
