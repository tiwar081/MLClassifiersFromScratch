import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io

# I write code to plot the isocontours of the following functions, each on its own separate figure.
# Make sure we can tell what isovalue each contour is associated with—you can do this with labels or a colorbar/legend.

def normal_dist(sigma, mu, x):
    A = 1/((2*np.pi)**len(sigma)*np.linalg.det(sigma))**0.5
    exp = -0.5 * np.matmul(np.matmul(np.transpose(x-mu), np.linalg.inv(sigma)), (x-mu))
    return A * np.e**exp

def normal_dist_plot(sigma, mu, x, y, title):
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            Z[i][j] = normal_dist(sigma, mu, np.array([X[i][j], Y[i][j]]))
    contour = plt.contour(X, Y, Z)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def normal_dist_diff_plot(sigma1, mu1, sigma2, mu2, x, y, title):
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            Z[i][j] = normal_dist(sigma1, mu1, np.array([X[i][j], Y[i][j]])) - normal_dist(sigma2, mu2, np.array([X[i][j], Y[i][j]]))
    contour = plt.contour(X, Y, Z)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

mu = np.array([1, 1])
sigma = np.array([[1, 0], [0, 2]])
x = np.linspace(-1, 3, 50)
y = np.linspace(-3, 5, 50)
title = "Function 1"
normal_dist_plot(sigma, mu, x, y, title)

mu = np.array([-1, 2])
sigma = np.array([[2, 1], [1, 4]])
x = np.linspace(-3, 1, 50)
y = np.linspace(-2, 6, 50)
title = "Function 2"
normal_dist_plot(sigma, mu, x, y, title)

mu1 = np.array([0, 2])
sigma1 = np.array([[2, 1], [1, 1]])
mu2 = np.array([2, 0])
sigma2 = sigma1
x = np.linspace(-3, 5, 50)
y = np.linspace(-2, 4, 50)
title = "Function 3"
normal_dist_diff_plot(sigma1, mu1, sigma2, mu2, x, y, title)

mu1 = np.array([0, 2])
sigma1 = np.array([[2, 1], [1, 1]])
mu2 = np.array([2, 0])
sigma2 = np.array([[2, 1], [1, 4]])
x = np.linspace(-5, 5, 100)
y = np.linspace(-3, 4, 100)
title = "Function 4"
normal_dist_diff_plot(sigma1, mu1, sigma2, mu2, x, y, title)

mu1 = np.array([1, 1])
sigma1 = np.array([[2, 0], [0, 1]])
mu2 = np.array([-1, -1])
sigma2 = np.array([[2, 1], [1, 2]])
x = np.linspace(-4, 4, 100)
y = np.linspace(-3, 3, 100)
title = "Function 5"
normal_dist_diff_plot(sigma1, mu1, sigma2, mu2, x, y, title)
