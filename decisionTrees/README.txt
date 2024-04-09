In this project, I implement a decision tree class! I tested the decision trees on SPAM and TITANIC datasets.

self.max_depth: depth of the tree.
self.data: data.
self.classes: binary classification of data.
self.feature_labels: feature labels.
self.fit(m): fits decision tree. m is number of features to be randomly sampled from at each node, 0 if using all features.
self.predict(Z): predicts on each row of Z.

Comments separate different sections of the python document.
In particular, there are sections that divide the code into decision tree modeling for SPAM and TITANIC dataset, and further subdivide into code for validation and for Kaggle.