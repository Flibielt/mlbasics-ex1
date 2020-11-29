import numpy as np
from .compute_cost_multi import compute_cost_multi


def gradient_descent_multi(x, y, theta, alpha, num_iters):
    """
    gradient_descent_multi Performs gradient descent to learn theta
        theta = gradient_descent_multi(x, y, theta, alpha, num_iters) updates theta by
        taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    m = len(y)  # number of training examples
    j_history = np.zeros((num_iters, 1))

    for iteration in range(num_iters):
        """
        ====================== YOUR CODE HERE ======================
        Instructions: Perform a single gradient step on the parameter vector theta.
         
        Hint: While debugging, it can be useful to print out the values
        of the cost function (computeCostMulti) and gradient here.
        """

        # ============================================================

        # Save the cost j in every iteration
        j_history[iteration] = compute_cost_multi(x, y, theta)

    return theta
