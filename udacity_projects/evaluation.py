#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.3, random_state=42)


### it's all yours from here forward!  

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
acc = clf.score(features_test, labels_test)
test_pred = clf.predict(features_test)
print 'Accuracy:', acc
print 'Num POIs predicted in test set: ', sum(clf.predict(features_test))
print 'Test set size: ', len(features_test)
print confusion_matrix(labels_test, test_pred)
print precision_recall_fscore_support(labels_test, test_pred)

# sample data for lesson 14 exercises
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print confusion_matrix(true_labels, predictions)
