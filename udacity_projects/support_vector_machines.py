#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
# your code goes here #
clf = SVC(C=10000, kernel='rbf')

# now your job is to fit the classifier
# using the training features/labels, and to
# make a set of predictions on the test data

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

# store your predictions in a list named pred

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

acc = accuracy_score(pred, labels_test)
print 'accuracy: ', acc

n_chris = sum(pred)
print 'n chris: ', n_chris

print 'term 10: ', pred[10]
print 'term 26: ', pred[26]
print 'term 50: ', pred[50]

#########################################################
