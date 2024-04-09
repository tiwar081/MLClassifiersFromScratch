General Notes:
Ensure that datafiles are in the same directory as the scripts folder before running.

IsocontourExploration.py and RandGaussianExploration.py:
These files are exploratory (and slightly disorganized). The entire file can be run at once! Each relevant graph will appear one by one.

LDA_QDA_Classification.py:
In this document, I hard code LDA and QDA. The MNIST dataset is fit by 10 gaussians, the spam dataset is fit by 2. For LDA, the gaussians are assumed to be distributed with the same covariance. Then, the highest likelihood gaussians are predicted for future data.
Sections are bounded by triple-hash comments.
To run any section, comment out all others.
