import os
import numpy as np
from matplotlib import pyplot

from .utils import pause, feature_normalize, gradient_descent_multi, normal_eqn


def ex1_multi():
    """
    Exercise 1: Linear regression with multiple variables

    Instructions
    ------------

    This file contains code that helps you get started on the
    linear regression exercise.

    You will need to complete the following functions in this
    exercises:

        warm_up_exercise.py
        plot_data.py
        gradient_descent.py
        compute_cost.py
        gradient_descent_multi.py
        compute_cost_multi.py
        feature_normalize.py
        normal_eqn.py

    For this part of the exercise, you will need to change some
    parts of the code below for various experiments (e.g., changing
    learning rates).
    """

    """
    ================ Part 1: Feature Normalization ================
    """
    # Load data
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/data/ex1data2.txt'
    file_path = file_path.replace('\\', '/')
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    m = y.size

    # print out some data points
    print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
    print('-' * 26)
    for i in range(10):
        print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))

    pause()

    print('Normalizing Features ...\n')
    # call featureNormalize on the loaded data
    X_norm, mu, sigma = feature_normalize(X)

    print('Computed mean:', mu)
    print('Computed standard deviation:', sigma)

    # Add intercept term to X
    X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

    """
    ================ Part 2: Gradient Descent ================

    ====================== YOUR CODE HERE ======================
    Instructions: We have provided you with the following starter
                  code that runs gradient descent with a particular
                  learning rate (alpha). 
    
                  Your task is to first make sure that your functions - 
                  computeCost and gradientDescent already work with 
                  this starter code and support multiple variables.
    
                  After that, try running gradient descent with 
                  different values of alpha and see which one gives
                  you the best result.
    
                  Finally, you should complete the code at the end
                  to predict the price of a 1650 sq-ft, 3 br house.
    
    Hint: At prediction, make sure you do the same feature normalization.
    """

    print('Running gradient descent ...')

    # Choose some alpha value - change this
    alpha = 0.1
    num_iters = 400

    # init theta and run gradient descent
    theta = np.zeros(3)
    theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    fig = pyplot.figure()
    pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Cost J')

    pyplot.show()

    # Display the gradient descent's result
    print('theta computed from gradient descent: {:s}'.format(str(theta)))

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ======================= YOUR CODE HERE ===========================
    # Recall that the first column of X is all-ones.
    # Thus, it does not need to be normalized.

    price = 0   # You should change this

    # ===================================================================

    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))

    pause()

    """
    ================ Part 3: Normal Equations ================
    """

    print('Solving with normal equations...')

    """
    ====================== YOUR CODE HERE ======================
    Instructions: The following code computes the closed form 
                  solution for linear regression using the normal
                  equations. You should complete the code in 
                  normal_eqn.py
    
                  After doing so, you should complete this code 
                  to predict the price of a 1650 sq-ft, 3 br house.
    """

    # Load data
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    m = y.size

    # Add intercept term to X
    X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

    # Calculate the parameters from the normal equation
    theta = normal_eqn(X, y)

    # Display normal equation's result
    print('Theta computed from the normal equations: {:s}'.format(str(theta)))

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ====================== YOUR CODE HERE ======================

    price = 0  # You should change this

    # ============================================================

    print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))
