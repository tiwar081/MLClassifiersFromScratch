import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
import random

random.seed(42)

# In this program, I explore a sample of data collected from a multivariate gaussian.

n = 100
mu1 = 3
sigma1 = 3
mu2 = 4
sigma2 = 2
sample = np.zeros((100, 2))
for i in range(100):
    x1 = random.normalvariate(mu1, sigma1)
    x2 = (x1/2 + random.normalvariate(mu2, sigma2))
    sample[i] = np.array([x1, x2])

sample = np.transpose(sample)
mu = np.mean(sample, axis=1)
#sample mean
print("Sample Mean:", mu)

#sample covariance
sig = np.cov(sample)
print("Sample Cov:", sig)

#eigenvalues and eigenvectors
evals, evects = np.linalg.eig(sig)
print("First eigenvalue-eigenvector pair:", evals[0], evects[:, 0])
print("Second eigenvalue-eigenvector pair:", evals[1], evects[:, 1])

#[-15, 15]
# (i) Plot all n = 100 data points
# (ii) arrows representing both covariance eigenvectors.
# The eigenvector arrows should originate at the mean and have magnitudes equal to their corresponding eigenvalues.

plt.scatter(sample[0], sample[1])
plt.xlim(-15, 15)
plt.ylim(-15, 15)
x, y = mu
dx1 = evals[0] * evects[:, 0]
dx2 = evals[1] * evects[:, 1]
plt.arrow(x, y, dx1[0], dx1[1], length_includes_head=True, head_width=0.2, head_length=0.3, fc='red')
plt.arrow(x, y, dx2[0], dx2[1], length_includes_head=True, head_width=0.2, head_length=0.3, fc='red')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("X1 ~ N(3,9)")
plt.ylabel("X2 ~ 1/2 X1 + N(4,4)")
plt.title("Gaussian Distribution of Two Random Variables")
plt.show()

# Rotate scatter so that eigenvectors are axis-aligned.
rotated_sample = np.zeros((2, 100))
for i in range(100):
    rotated_sample[:, i] = evects.T @ (sample[:, i] - mu)
plt.scatter(rotated_sample[0], rotated_sample[1])
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("X1 ~ N(3,9)")
plt.ylabel("X2 ~ 1/2 X1 + N(4,4)")
plt.title("Centered and Rotated Gaussian Distribution of Two Random Variables")
plt.show()
