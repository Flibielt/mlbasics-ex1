from .utils import warm_up_exercise, pause, plot_data, compute_cost, gradient_descent
import os
import numpy as np
from matplotlib import pyplot


def ex1():
    """
    Machine Learning Online Class - Exercise 1: Linear Regression

    Instructions
    ------------

    This file contains code that helps you get started on the
    linear exercise. You will need to complete the following functions
    in this exercises:

        warm_up_exercise.py
        plot_data.py
        gradient_descent.py
        compute_cost.py
        gradient_descent_multi.py
        compute_cost_multi.py
        feature_normalize.py
        normal_eqn.py

    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.

    x refers to the population size in 10,000s
    y refers to the profit in $10,000s
    """

    """
    ==================== Part 1: Basic Function ====================
    """
    # Complete warmUpExercise.py
    print('Running warmUpExercise ...')
    print('5x5 Identity Matrix:')
    warm_up_exercise()
    pause()

    """
    ======================= Part 2: Plotting =======================
    """
    print('Plotting Data ...')
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/data/ex1data1.txt'
    file_path = file_path.replace('\\', '/')
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, 0]
    y = data[:, 1]
    m = y.size

    # Plot data
    # Note: You have to complete the code in plot_data.py
    plot_data(X, y)
    pause()

    """
    =================== Part 3: Cost and Gradient descent ===================
    """

    X = np.stack([np.ones(m), X], axis=1)   # Add a column of ones to x
    theta = np.zeros(2)    # initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    print('\nTesting the cost function ...')
    # compute and display initial cost
    J = compute_cost(X, y, theta)
    print('With theta = [0 ; 0]\nCost computed = %.2f\n', J)
    print('Expected cost value (approx) 32.07')

    # further testing of the cost function
    J = compute_cost(X, y, [-1, 2])
    print('\nWith theta = [-1 ; 2]\nCost computed = %.2f\n', J)
    print('Expected cost value (approx) 54.24\n')

    pause()

    print('\nRunning Gradient Descent ...\n')
    # run gradient descent
    theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

    # print theta to screen
    print('Theta found by gradient descent:\n')
    print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
    print('Expected theta values (approx)\n')
    print(' -3.6303\n  1.1664\n\n')

    # Plot the linear fit
    plot_data(X[:, 1], y)

    pyplot.plot(X[:, 1], np.dot(X, theta), '-')
    pyplot.legend(['Training data', 'Linear regression'])
    pyplot.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.dot([1, 3.5], theta)
    print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1 * 10000))
    predict2 = np.dot([1, 7], theta)
    print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2 * 10000))

    pause()

    """
    ============= Part 4: Visualizing J(theta_0, theta_1) =============
    """

    print('Visualizing J(theta_0, theta_1) ...')

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

    # Fill out J_vals
    for i in range(len(theta0_vals)):
        for J in range(len(theta1_vals)):
            t = [theta0_vals[i], theta1_vals[J]]
            J_vals[i, J] = compute_cost(X, y, t)

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals.T

    # Surface plot
    fig = pyplot.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
    pyplot.xlabel('theta0')
    pyplot.ylabel('theta1')
    pyplot.title('Surface')

    # Contour plot
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    ax = pyplot.subplot(122)
    pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
    pyplot.xlabel('theta0')
    pyplot.ylabel('theta1')
    pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
    pyplot.title('Contour, showing minimum')
    pyplot.show()

