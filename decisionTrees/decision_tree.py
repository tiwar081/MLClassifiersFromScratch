"""
Have Fun!
- 189 Course Staff
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from pydot import graph_from_dot_data
import io
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number

class DecisionTree:
    def __init__(self, max_depth=3, classes=None, data=None, feature_labels=None):
        self.max_depth = max_depth
        self.classes = classes
        self.data = data
        self.features = feature_labels
        self.feature_indx = np.array(range(len(data[0])))
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.pred = None  # for leaf nodes
        self.leaf = False

    @staticmethod
    def entropy(y):
        y = np.array(y)
        # y is a np array
        n = len(y)
        entropy = 0
        for C in set(y):
            num_C = np.count_nonzero(y == C)
            p_C = num_C/n
            entropy += -1 * p_C * np.log(p_C)/np.log(2)
        return entropy
    
    @staticmethod
    def binary_entropy(numC1, numC2):
        n = numC1 + numC2
        if n == 0:
            return 0
        p_C1, p_C2 = numC1/n, numC2/n
        entropy = 0
        if p_C1 != 0:
            entropy += - p_C1 * np.log(p_C1)/np.log(2)
        if p_C2 != 0:
            entropy += - p_C2 * np.log(p_C2)/np.log(2)
        return entropy
    
    @staticmethod
    def binary_info_gain(C1l, C2l, C1r, C2r):
        nl, nr = C1l + C2l, C1r + C2r
        h = DecisionTree.binary_entropy(C1l + C1r, C2l + C2r)
        h_aft = (DecisionTree.binary_entropy(C1l, C2l)*nl + DecisionTree.binary_entropy(C1r, C2r)*nr)/(nl+nr)
        return h - h_aft

    @staticmethod
    def information_gain(X_col, y, thresh):
        X_col = np.array(X_col)
        # X_col is the column of features across which we are determining split, np array
        h = DecisionTree.entropy(y)
        y_r = y[X_col >= thresh]
        y_l = y[X_col < thresh]
        h_aft = (DecisionTree.entropy(y_l)*len(y_l) + DecisionTree.entropy(y_r)*len(y_r))/len(y)
        return h - h_aft
    
    # @staticmethod
    # def gini_impurity(X, y, thresh):
    #     # TODO
    #     pass

    # @staticmethod
    # def gini_purification(X, y, thresh):
    #     # TODO
    #     pass

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def find_thresh(self, X_col, y):
        # sorted indices
        iX_col_sorted = np.argsort(X_col)
        c0l, c1l = 0, 0
        c0r, c1r = len(y) - np.count_nonzero(y), np.count_nonzero(y)
        # choose best threshold
        best_thresh, best_info_gain = X_col[iX_col_sorted[0]], DecisionTree.binary_info_gain(c0l, c1l, c0r, c1r)
        thresh, info_gain = best_thresh, best_info_gain
        for j in iX_col_sorted[1:]:
            if y[j] == 0:
                c0l += 1
                c0r -= 1
            else:
                c1l += 1
                c1r -= 1
            # Want a unique threshold.
            if thresh == X_col[j]:
                continue
            thresh, info_gain = X_col[j], DecisionTree.binary_info_gain(c0l, c1l, c0r, c1r)
            if info_gain > best_info_gain:
                best_thresh, best_info_gain = thresh, info_gain
        return best_thresh, best_info_gain

    def grow(self, X, y, m=0):
        # Base Case
        if self.max_depth == 0 or np.all(y == y[0]):
            self.pred = stats.mode(y)[0]
            self.leaf = True
            return
        else:
            # Choose m features
            if m>0:
                self.feature_indx = np.random.choice(self.feature_indx, size=m, replace=False)
            # Choose Best Splitting feature
            thresholds = np.zeros(len(self.feature_indx))
            info_gains = np.zeros(len(self.feature_indx))
            for i, X_i in enumerate(self.feature_indx):
                X_col = X[:, X_i]
                thresholds[i], info_gains[i] = self.find_thresh(X_col, self.classes)
            i = np.argmax(info_gains)
            self.split_idx = self.feature_indx[i]
            self.thresh = thresholds[i]
            #Split
            Xl, yl, Xr, yr = self.split(X, y, self.split_idx, self.thresh)
            if len(Xl) == 0 or len(Xr) == 0:
                self.max_depth = 0
                self.grow(X, y)
                return
            self.left = DecisionTree(max_depth=self.max_depth-1, classes=yl, data=Xl, feature_labels=self.features)
            self.right = DecisionTree(max_depth=self.max_depth-1, classes=yr, data=Xr, feature_labels=self.features)
            self.left.grow(Xl, yl)
            self.right.grow(Xr, yr)
            return
    
    def fit(self, m=0):
        self.grow(self.data, self.classes, m)

    def predict_single(self, Z_row, show_path=False):
        if show_path==False:
            if self.leaf:
                return self.pred
            if Z_row[self.split_idx] >= self.thresh:
                return self.right.predict_single(Z_row)
            else:
                return self.left.predict_single(Z_row)
        else:
            if self.leaf:
                print(self.pred)
                return self.pred
            if Z_row[self.split_idx] >= self.thresh:
                print(self.features[self.split_idx], '>=', self.thresh)
                return self.right.predict_single(Z_row, show_path=True)
            else:
                print(self.features[self.split_idx], '<', self.thresh)
                return self.left.predict_single(Z_row, show_path=True)

    def predict(self, Z):
        # Returns predictions for every entry in data.
        predictions = np.zeros(len(Z))
        for i, entry in enumerate(Z):
            predictions[i] = self.predict_single(entry)
        return predictions

    def __repr__(self):
        if self.leaf == True:
            return "%s (%s)" % (self.pred, self.classes.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())

# Replace with own decision tree
class BaggedTrees():
    def __init__(self, max_depth=3, classes=None, data=None, feature_labels=None, n=200):
        self.n = n
        self.max_depth=max_depth
        self.classes=classes
        self.data=data
        self.feature_labels=feature_labels
        self.decision_trees = []
        for i in range(self.n):
            X, y = BaggedTrees.bagged(data, classes)
            self.decision_trees.append(DecisionTree(max_depth=max_depth, classes=y, data=X, feature_labels=feature_labels))

    @staticmethod
    def bagged(X, y):
        num_rows = len(X)
        sampled_indices = np.random.choice(num_rows, size=num_rows, replace=True)
        X_r = X[sampled_indices]
        y_r = y[sampled_indices]
        return X_r, y_r

    def fit(self):
        for tree in self.decision_trees:
            tree.fit()

    def predict_single(self, Z_row):
        predictions = [tree.predict_single(Z_row) for tree in self.decision_trees]
        return stats.mode(predictions)[0]

    def predict(self, Z):
        # Returns predictions for every entry in data.
        predictions = self.decision_trees[0].predict(Z)
        for tree in self.decision_trees[1:]:
            predictions += tree.predict(Z)
        return np.round(predictions/self.n)


class RandomForest(BaggedTrees):
    def __init__(self, max_depth=3, classes=None, data=None, feature_labels=None, n=200, m=1):
        self.m = m
        super().__init__(max_depth=max_depth, classes=classes, data=data, feature_labels=feature_labels, n=n)
    
    def fit(self):
        for tree in self.decision_trees:
            tree.fit(m=self.m)


# class BoostedRandomForest(RandomForest):

#     def fit(self, X, y):
#         # TODO
#         pass
    
#     def predict(self, X):
#         # TODO
#         pass


def mode_exc_n1(data_col):
    mc = Counter(data_col).most_common(2)
    mode = mc[0][0]
    if mode == -1:
        mode = mc[1][0]
    return mode


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for j in range(len(data[0])):
            m = mode_exc_n1(data[:, j])
            for i in range(len(data)):
                if data[i][j] == -1:
                    data[i][j] = m
    
    return data, onehot_features


def evaluate(clf):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


if __name__ == "__main__":
    # dataset = "titanic"
    dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=float).astype(int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)
    
    ### TITANIC (Validation) ###
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # clf = DecisionTree(classes=y_train, data=X_train, feature_labels=features, max_depth=5)
    # clf.fit()
    # y_pred = clf.predict(X_train)
    # print("Training DT Titanic:", np.mean(y_pred == y_train))
    # y_pred = clf.predict(X_val)
    # print("Validation DT Titanic:", np.mean(y_pred == y_val))

    # clf = RandomForest(classes=y_train, data=X_train, feature_labels=features, max_depth=8, n=500, m=4)
    # clf.fit()
    # y_pred = clf.predict(X_train)
    # print("Training RF Titanic:", np.mean(y_pred == y_train))
    # y_pred = clf.predict(X_val)
    # print("Validation RF Titanic:", np.mean(y_pred == y_val))
    ### TITANIC (Validation) ###

    ### SPAM (Validation) ###
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # clf = DecisionTree(classes=y_train, data=X_train, feature_labels=features, max_depth=5)
    # clf.fit()
    # y_pred = clf.predict(X_train)
    # print("Training DT Spam:", np.mean(y_pred == y_train))
    # y_pred = clf.predict(X_val)
    # print("Validation DT Spam:", np.mean(y_pred == y_val))

    # clf = RandomForest(classes=y_train, data=X_train, feature_labels=features, max_depth=8, n=400, m=5)
    # clf.fit()
    # y_pred = clf.predict(X_train)
    # print("Training RF Spam:", np.mean(y_pred == y_train))
    # y_pred = clf.predict(X_val)
    # print("Validation RF Spam:", np.mean(y_pred == y_val))
    ### SPAM (Validation) ###

    # def results_to_csv(y_test):
    #     y_test = y_test.astype(int)
    #     df = pd.DataFrame({'Category': y_test})
    #     df.index += 1 # Ensures that the index starts at 1
    #     df.to_csv('submission.csv', index_label='Id')

    # ### SPAM (Kaggle) ###
    # # clf = RandomForest(classes=y, data=X, feature_labels=features, max_depth=8, n=400, m=5)
    # ### TITANIC (Kaggle) ###
    # clf = RandomForest(classes=y, data=X, feature_labels=features, max_depth=8, n=500, m=4)
    
    # ### Kaggle ###
    # clf.fit()
    # y_pred = clf.predict(Z)
    # results_to_csv(y_pred)

    # clf = DecisionTree(classes=y, data=X, feature_labels=features, max_depth=5)
    # clf.fit()
    # clf.predict_single(Z[0], show_path=True)
    # clf.predict_single(Z[10], show_path=True)

    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # accuracies = []
    # for md in np.linspace(1, 40, 20):
    #     clf = DecisionTree(classes=y_train, data=X_train, feature_labels=features, max_depth=md)
    #     clf.fit()
    #     y_pred = clf.predict(X_val)
    #     accuracies.append(np.mean(y_pred == y_val))
    # plt.plot(np.linspace(1, 40, 20), accuracies)
    # plt.xlabel("depth")
    # plt.ylabel("accuracies")
    # plt.title("Validation Accuracy Across Varying Max Depth")
    # plt.show()