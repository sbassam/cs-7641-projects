import algorithms
import loader

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns

from algorithms import ann
from plotting import plot_learning_curve, plot_validation_curve

iris = datasets.load_iris()
digits = datasets.load_digits()
wine = datasets.load_wine()
breast = datasets.load_breast_cancer()
abalone = loader.process_abalone()
from sklearn.model_selection import train_test_split
import math
import matplotlib as mplt

#iris = digits
#iris = abalone #good
iris = wine






features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.3)


# scaling used https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
scaler = StandardScaler()
scaler.fit(features_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)



#
# print('##################### Decision tree ######################')
#
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# tree_score = clf.score(features_test, labels_test)
# print(tree_score)
#
#
# tree_precision = precision_score(labels_test, pred, average='weighted')
# print('precision for tree: ', tree_precision)
# tree_recall = recall_score(labels_test, pred, average='weighted')
# print('recall for tree: ', tree_recall)
#
#
# # plotting
# # dot_data = tree.export_graphviz(clf, out_file=None)
# # graph = graphviz.Source(dot_data)
# # graph.render("iris-")
# #
# # dot_data = tree.export_graphviz(clf, out_file=None,
# #                                 feature_names=iris.feature_names,
# #                                 class_names=iris.target_names,
# #                                 filled=True,
# #                                 rounded=True,
# #                                 special_characters=True)
# # graph = graphviz.Source(dot_data)
# plot_learning_curve(clf, features_train, labels_train, 'Decision-Tree', np.linspace(0.01, 1.0, 10), 8)

# print('##################### SVM ######################')
# from sklearn.svm import SVC
# clf = SVC()
#
# # fit:
# clf.fit(features_train, labels_train)
#
# # predict:
# pred = clf.predict(features_test)
#
# # score:
# svm_score = clf.score(features_test, labels_test)
# print(svm_score)
# SVM_precision = precision_score(labels_test, pred, average='macro')
# print('precision for SVM: ', SVM_precision)
# SVM_recall = recall_score(labels_test, pred, average='weighted')
# print('recall for SVM: ', SVM_recall)
# plot_learning_curve(clf, features_train, labels_train, 'SVM', np.linspace(0.01, 1.0, 10), 8)


print('##################### Neural Network ######################')
# abalone
#hidden_layer_size_arr_ax = np.arange(32, 52, 4)
#hidden_layer_size_arr = np.array([(i, i, i) for i in hidden_layer_size_arr_ax])

#num_hidden_layers_arr = np.array([(40, 40), (40, 40, 40), (40, 40, 40, 40), (40, 40, 40, 40, 40), (40, 40, 40, 40, 40, 40)])
#num_hidden_layers_arr_ax = np.arange(2, 7, 1)

#learning_rate = 0
#momentum = 0

# wine
hidden_layer_size_arr_ax = np.arange(1, 20, 4)
hidden_layer_size_arr = np.array([(i, i, i) for i in hidden_layer_size_arr_ax])

num_hidden_layers_arr = np.array([(6), (6, 6), (6, 6, 6), (6, 6, 6, 6), (6, 6, 6, 6, 6)])
num_hidden_layers_arr_ax = np.arange(1, 6, 1)

learning_rate = 0
momentum = 0

clf = ann.setup_ann(features_train, labels_train, features_test, labels_test)
#plot_learning_curve(clf, features_train, labels_train, 'Neural-Network', np.linspace(0.01, 1.0, 10), 8)
plot_validation_curve(clf, features_train, labels_train, 'Neural-Network', 'hidden_layer_sizes',
                      hidden_layer_size_arr, hidden_layer_size_arr_ax)
plot_validation_curve(clf, features_train, labels_train, 'Neural-Network', 'hidden_layer_sizes',
                      num_hidden_layers_arr, num_hidden_layers_arr_ax)
ann.fit_ann(clf, features_train, labels_train)
pred = ann.predict_ann(clf, features_test)
ann_precision, ann_recall, ann_f1 = ann.get_performance_ann(clf, pred, features_test, labels_test)



print('##################### Boosting ######################')


# print('##################### kNN ######################')
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(3, weights='uniform')
# clf.fit(features_train, labels_train)
#
# # predict:
# pred = clf.predict(features_test)
#
# # score:
# kNN_score = clf.score(features_test, labels_test)
# print(kNN_score)
#
# kNN_precision = precision_score(labels_test, pred, average='micro')
# print('precision for kNN: ', kNN_precision)
# kNN_recall = recall_score(labels_test, pred, average='weighted')
# print('recall for kNN: ', kNN_recall)
# plot_learning_curve(clf, features_train, labels_train, 'kNN', np.linspace(0.01, 1.0, 10), 8)













