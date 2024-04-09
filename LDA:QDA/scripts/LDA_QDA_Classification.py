import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import scipy.linalg as la
import pandas as pd
import random as rand

from save_csv import results_to_csv

rand.seed(21)

#Helpers
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
    plt.show()

def l2_norm(arr):
    cum = 0
    for val in arr:
        cum += val**2
    return cum**0.5

def l2_normalize(arr):
    return np.array(arr)/l2_norm(arr)

def accuracy(true_labels, predicted_labels):
    if len(true_labels) != len(predicted_labels):
        print("ERROR: true_labels and predicted_labels must be the same length")
        exit()
    return sum(np.array(true_labels)==np.array(predicted_labels))/len(true_labels)

def MNIST_validation():
    MNIST = dict(np.load(f"../data/mnist-data-hw3.npz"))
    MNIST["training_data"] = MNIST["training_data"].reshape(len(MNIST["training_data"]), -1)
    n_split = 10000
    indices = list(range(len(MNIST["training_data"])))
    rand.shuffle(indices)
    # Use shuffled indices to split data and labels
    validation_indices = indices[:n_split]
    training_indices = indices[n_split:]
    # Assign data and labels to validation and training sets
    MNIST["validation_set"] = MNIST["training_data"][validation_indices]
    MNIST["training_data"] = MNIST["training_data"][training_indices]
    MNIST["validation_labels"] = MNIST["training_labels"][validation_indices]
    MNIST["training_labels"] = MNIST["training_labels"][training_indices]
    return MNIST

def normalize(training_data, training_labels):
    """
    Returns pandas df of normalized and sorted data.
    """
    data = pd.DataFrame(training_data)

    #l2 normalization
    #Makes each row of the dataframe a numpy array
    data_normalized = data.apply(l2_normalize, axis=1, result_type='expand')

    data_normalized['labels'] = training_labels
    return data_normalized

def evaluate_MNIST_gaussian(training_data, training_labels):
    """
    Returns 10-length arrays of sigma, mu, n.
    """
    data_normalized = normalize(training_data, training_labels)

    sigma = []
    mu = []
    n = []
    for i in range(10):
        class_data = data_normalized.loc[data_normalized["labels"] == i, data_normalized.columns!="labels"]
        sigma.append(class_data.cov())
        mu.append(class_data.mean())
        n.append(len(class_data))

    return np.array(sigma), np.array(mu), np.array(n)

# ## Testing methods for fitting a gaussian to the data (1). Then, visualizing covariance matrix (2) ###
# dat = MNIST_validation()
# training_data = dat["training_data"]
# training_labels = dat["training_labels"]
# validation_set = dat["validation_set"]
# validation_labels = dat["validation_labels"]

# #1
# #separate data by digit
# #fit mean and covariance matrices for each class
# sigma, mu, n = evaluate_MNIST_gaussian(training_data, training_labels)

# #2, Visualizing the covariance matrix
# digit = 5
# plt.imshow(sigma[digit])
# plt.colorbar()
# plt.show()
# ## END ###

#LDA
def classify_lda(B, mu, n, x):
    """
    B: array of (muC.T @ sigma^-1).T
    mu: array of vector means for each class
    n: array of point counts in each class
    x: points to be classified
    returns digit classifications of points in x
    """
    predictions = np.zeros(len(x))
    N = np.sum(n)
    for i in range(len(predictions)):
        rel_class_probs = np.zeros(10)
        for C in range(10):
            rel_class_probs[C] = np.dot(B[C], x[i]) - np.dot(B[C], mu[C])/2 + np.log(n[C]/N)
        predictions[i] = np.argmax(rel_class_probs)
    return predictions

def B_LDA(LDA_cov, mu):
    #LDA_cov @ B[C] = mu[C]
    numC = len(mu)
    len_pts = len(mu[0])
    B = np.zeros((numC, len_pts))
    epsilon = 1e-15
    for C in range(numC):
        cov = LDA_cov + epsilon * np.eye(len_pts)
        B[C] = np.linalg.solve(cov, mu[C])
    return B

def calculate_LDA_cov(sigma, n):
    total = np.zeros(sigma[0].shape)
    for i in range(len(sigma)):
        total += sigma[i] * n[i]
    return total/sum(n)

def train_MNIST_LDA_models():
    """
    task: train model with raw pixels as features.
    Use num_training_examples to train various models, evaluate with validation set.
    """
    num_training_examples = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
    MNIST = MNIST_validation()
    # train_err = []
    val_err = []
    for samples in num_training_examples:
        # Train model and run it on training and validation sets.
        training_data = MNIST["training_data"][:samples]
        training_labels = MNIST["training_labels"][:samples]
        sigma, mu, n = evaluate_MNIST_gaussian(training_data, training_labels)
        LDA_cov = calculate_LDA_cov(sigma, n)
        B = B_LDA(LDA_cov, mu)
        # train_pred = classify_lda(B, mu, n, training_data)
        val_pred = classify_lda(B, mu, n, MNIST["validation_set"])
        #Evaluate and plot accuracies.
        # train_err.append(1-accuracy(train_pred, training_labels))
        val_err.append(1-accuracy(val_pred, MNIST["validation_labels"]))

    return num_training_examples, val_err

### START LDA ###
num_training_examples, val_err = train_MNIST_LDA_models()
# plt.plot(num_training_examples, train_err, label="Training Error")
plt.plot(num_training_examples, val_err, label="Validation Error")
plt.xlabel("Number of Training Examples")
plt.ylabel("Error")
plt.title("Validation Errors of LDA models of MNIST data")
plt.legend()
plt.show()
### END LDA ###

#QDA
def classify_QDA(sigma, mu, n, x):
    """
    sigma: array of covariance matrices
    mu: array of vector means for each class
    n: array of point counts in each class
    x: points to be classified
    returns digit classifications of points in x
    """
    D = np.eye(784)
    predictions = np.zeros(len(x))
    N = np.sum(n)
    for i in range(len(predictions)):
        rel_class_probs = np.zeros(10)
        for C in range(10):
            rel_class_probs[C] = pdf_QDA(sigma[C], mu[C], n[C]/N, x[i], D)
        predictions[i] = np.argmax(rel_class_probs)
    return predictions

def pdf_QDA(sigma, mu, prior, x, diag):
    """
    sigma, log_det, mu, and prior are all class specific.
    returns relative probability of x being in a certain class
    """
    epsilon = 1e-15
    cov = sigma + epsilon * diag
    a = np.linalg.solve(cov, x - mu)
    epsilon = 1
    cov = sigma + epsilon * diag
    b = la.det(cov)
    if b > 1e-20:
        b = np.log(b)
    else:
        b = 0
    return -0.5*np.dot(a, x-mu) - 0.5*b + np.log(prior)

def train_MNIST_QDA_models():
    """
    task: train model with raw pixels as features.
    Use num_training_examples to train various models, evaluate with validation set.
    """
    # num_training_examples = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
    num_training_examples = [100]
    MNIST = MNIST_validation()
    # train_err = []
    val_err = []
    for samples in num_training_examples:
        # Train model and run it on training and validation sets.
        training_data = MNIST["training_data"][:samples]
        training_labels = MNIST["training_labels"][:samples]
        sigma, mu, n = evaluate_MNIST_gaussian(training_data, training_labels)
        # train_pred = classify_QDA(sigma, mu, n, training_data)
        val_pred = classify_QDA(sigma, mu, n, MNIST["validation_set"])
        #Evaluate and plot accuracies.
        # train_err.append(1-accuracy(train_pred, training_labels))
        val_err.append(1-accuracy(val_pred, MNIST["validation_labels"]))
    return num_training_examples, val_err
    # return num_training_examples, train_err

### START QDA ###
# num_training_examples, val_err = train_MNIST_QDA_models()
# # num_training_examples, train_err = train_MNIST_QDA_models()
# # plt.plot(num_training_examples, train_err, label="Training Error")
# plt.plot(num_training_examples, val_err, label="Validation Error")
# plt.xlabel("Number of Training Examples")
# plt.ylabel("Error")
# plt.title("Training and Validation Errors of LDA models of MNIST data")
# plt.legend()
# plt.show()
### END QDA ###

#(d)
def lda_error_rate_digitwise(B, mu, n, x, labels):
    """
    B: array of (muC.T @ sigma^-1).T
    mu: array of vector means for each class
    n: array of point counts in each class
    x: points to be classified
    labels: classification of points in x
    returns digit classification errors of points in x, separated by digit (len 10 array)
    """
    N = np.sum(n)
    count = len(x)
    num_incorrect = np.zeros(10)
    num_total = np.zeros(10)
    for i in range(count):
        curr_dig = labels[i]
        num_total[curr_dig] += 1
        rel_class_probs = np.zeros(10)
        for C in range(10):
            rel_class_probs[C] = np.dot(B[C], x[i]) - np.dot(B[C], mu[C])/2 + np.log(n[C]/N)
        if (np.argmax(rel_class_probs) != curr_dig):
            num_incorrect[curr_dig] += 1
    return num_incorrect/num_total

def digitwise_classification_plot():
    """
    task: train model with raw pixels as features.
    Use num_training_examples to train various models, evaluate with validation set.
    """
    num_training_examples = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
    MNIST = MNIST_validation()
    val_err = []
    for samples in num_training_examples:
        # Train model and run it on training and validation sets.
        training_data = MNIST["training_data"][:samples]
        training_labels = MNIST["training_labels"][:samples]
        sigma, mu, n = evaluate_MNIST_gaussian(training_data, training_labels)
        LDA_cov = calculate_LDA_cov(sigma, n)
        B = B_LDA(LDA_cov, mu)
        val_err.append(lda_error_rate_digitwise(B, mu, n, MNIST["validation_set"], MNIST["validation_labels"]))
    val_err = np.transpose(np.array(val_err))
    return num_training_examples, val_err

### START LDA classification digitwise ###
# num_training_examples, val_err = digitwise_classification_plot()
# for i in range(10):
#     plt.plot(num_training_examples, val_err[i], label=f"Digit {i}")
# plt.xlabel("# of training points")
# plt.ylabel("Error Rate")
# plt.title("LDA digitwise classification")
# plt.legend()
# plt.show()
### END LDA classification digitwise ###

def train_MNIST_LDA_model():
    """
    task: train model with raw pixels as features.
    Use num_training_examples to train various models, evaluate with validation set.
    """
    MNIST = MNIST_validation()
    # Train model and run it on training and validation sets.
    training_data = MNIST["training_data"]
    training_labels = MNIST["training_labels"]
    sigma, mu, n = evaluate_MNIST_gaussian(training_data, training_labels)
    LDA_cov = calculate_LDA_cov(sigma, n)
    B = B_LDA(LDA_cov, mu)
    test_pred = classify_lda(B, mu, n, MNIST["test_data"].reshape(len(MNIST["test_data"]), -1))
    return test_pred

### START MNIST Kaggle Submission ###
# results_to_csv(train_MNIST_LDA_model())
### END MNIST Kaggle Submission ###


def classify_lda_spam(B, mu, n, x):
    """
    B: array of (muC.T @ sigma^-1).T
    mu: array of vector means for each class
    n: array of point counts in each class
    x: points to be classified
    returns spam or ham classifications of points in x
    """
    predictions = np.zeros(len(x))
    N = np.sum(n)
    for i in range(len(predictions)):
        rel_class_probs = np.zeros(2)
        for C in range(2):
            rel_class_probs[C] = np.dot(B[C], x[i]) - np.dot(B[C], mu[C])/2 + np.log(n[C]/N)
        predictions[i] = np.argmax(rel_class_probs)
    return predictions

def evaluate_spam_gaussian(training_data, training_labels):
    """
    Returns 2-length arrays of sigma, mu, n.
    """
    data = pd.DataFrame(training_data)
    data['labels'] = training_labels
    sigma = []
    mu = []
    n = []
    for i in range(2):
        class_data = data.loc[data["labels"] == i, data.columns!="labels"]
        sigma.append(class_data.cov())
        mu.append(class_data.mean())
        n.append(len(class_data))

    return np.array(sigma), np.array(mu), np.array(n)

def train_spam_LDA_model():
    """
    task: train model with features specified by generate_feature_vector() in featurize.py.
    Use num_training_examples to train various models, evaluate with validation set.
    """
    spam = dict(np.load("../data/spam-data-hw3.npz"))
    # Train model and run it on training and validation sets.
    training_data = spam["training_data"]
    training_labels = spam["training_labels"]
    sigma, mu, n = evaluate_spam_gaussian(training_data, training_labels)
    LDA_cov = calculate_LDA_cov(sigma, n)
    B = B_LDA(LDA_cov, mu)
    test_pred = classify_lda_spam(B, mu, n, spam["test_data"])
    return test_pred

### START SPAM Kaggle Submission ###
# results_to_csv(train_spam_LDA_model())
### END SPAM Kaggle Submission ###
