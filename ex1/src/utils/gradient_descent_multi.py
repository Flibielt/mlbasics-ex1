import numpy as np
from .compute_cost_multi import compute_cost_multi


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    """
    gradient_descent_multi Performs gradient descent to learn theta
        theta = gradient_descent_multi(x, y, theta, alpha, num_iters) updates theta by
        taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()

    J_history = []

    for iteration in range(num_iters):
        """
        ====================== YOUR CODE HERE ======================
        Instructions: Perform a single gradient step on the parameter vector theta.
         
        Hint: While debugging, it can be useful to print out the values
        of the cost function (compute_cost_multi) and gradient here.
        """

        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)

        # ============================================================

        # Save the cost j in every iteration
        J_history.append(compute_cost_multi(X, y, theta))

    return theta, J_history
