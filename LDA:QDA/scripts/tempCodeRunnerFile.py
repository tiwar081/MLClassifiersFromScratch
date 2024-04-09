## START PROBLEMS 1 AND 2 ###
dat = MNIST_validation()
training_data = dat["training_data"]
training_labels = dat["training_labels"]
validation_set = dat["validation_set"]
validation_labels = dat["validation_labels"]

#1
#separate data by digit
#fit mean and covariance matrices for each class
sigma, mu, n = evaluate_MNIST_gaussian(training_data, training_labels)

#2, Visualizing the covariance matrix
digit = 5
plt.imshow(sigma[digit])
plt.colorbar()
plt.show()
## END PROBLEMS 1 AND 2 ###