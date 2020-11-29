import numpy as np
from .compute_cost import compute_cost


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    gradient_descent Performs gradient descent to learn theta
        gradient_descent(X, y, theta, alpha, num_iters) updates theta by
        taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    J_history = [] # Use a python list to save cost in every iteration

    for iteration in range(num_iters):
        """
        ====================== YOUR CODE HERE ======================
        Instructions: Perform a single gradient step on the parameter vector theta. 
        
        Hint: While debugging, it can be useful to print out the values
        of the cost function (compute_cost) and gradient here.
        """

        # ============================================================
        # Save the cost j in every iteration
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history
