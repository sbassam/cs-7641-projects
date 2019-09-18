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

from algorithms import ann, knn, dt
from plotting import plot_learning_curve, plot_validation_curve

iris = datasets.load_iris()
digits = datasets.load_digits()
wine = datasets.load_wine()
breast = datasets.load_breast_cancer()
abalone = loader.process_abalone()
abalone_ternary = loader.process_abalone_ternary()
white_wine_quality = loader.process_white_wine_quality()
red_wine_quality = loader.process_red_wine_quality()
wine_quality = loader.process_wine_quality()
from sklearn.model_selection import train_test_split
import math
import matplotlib as mplt

# iris = digits
# iris = abalone #good
iris = wine_quality
ds_name = 'wine_quality'


def log_performance(dataset_name, learner, score, precision, recall, f1):
    """
    write the learner's performance measures on a specific dataset to a file.
    :param dataset_name: name of the dataset
    :param learner: name of the learner
    :param score: learner score on the testing set
    :param precision: learner precision on the testing set
    :param recall: learner recall on the testing set
    :param f1: learner f1 on the testing set
    :return:
    """
    f = open("performance_results.txt", "a")

    statement = "\n----------Performance Summary----------\n" \
                "   Dataset: %s\n" \
                "   Learner: %s\n" \
                "Score: %s\n" \
                "Precision: %s\n" \
                "Recall: %s\n" \
                "F1: %s\n" \
                "\n--------------------------------------------\n"
    f.write(statement % (dataset_name, learner, score, precision, recall, f1))
    f.close()

    return


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

# n_neighbors_arr = np.arange(1, 100, 5)
max_depth_arr = np.arange(1, 50, 5)
#
#
clf = dt.setup_dt(features_train, labels_train, features_test, labels_test)
#
# plot learning curve
plot_learning_curve(clf, features_train, labels_train, 'Decision Tree', ds_name, np.linspace(.1, 1., 10), 8)

# plot validation curve
plot_validation_curve(clf, features_train, labels_train, 'Decision Tree', ds_name, 'max_depth', max_depth_arr)

# fit

dt.fit_dt(clf, features_train, labels_train)

# predict testing set
pred = dt.predict_dt(clf, features_test)

dt_score, dt_precision, dt_recall, dt_f1 = dt.get_performance_dt(clf, pred, features_test, labels_test)

log_performance(ds_name, "decision tree", dt_score, dt_precision, dt_recall, dt_f1)

# plotting
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris-")
#
# dot_data = tree.export_graphviz(clf, out_file=None,
#                                 feature_names=iris.feature_names,
#                                 class_names=iris.target_names,
#                                 filled=True,
#                                 rounded=True,
#                                 special_characters=True)
# graph = graphviz.Source(dot_data)

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
# hidden_layer_size_arr_ax = np.arange(32, 52, 4)
# hidden_layer_size_arr = np.array([(i, i, i) for i in hidden_layer_size_arr_ax])

# num_hidden_layers_arr = np.array([(40, 40), (40, 40, 40), (40, 40, 40, 40), (40, 40, 40, 40, 40), (40, 40, 40, 40, 40, 40)])
# num_hidden_layers_arr_ax = np.arange(2, 7, 1)

# learning_rate = 0
# momentum = 0

# wine
# hidden_layer_size_arr_ax = np.arange(1, 20, 4)
# hidden_layer_size_arr = np.array([(i, i, i) for i in hidden_layer_size_arr_ax])
#
# num_hidden_layers_arr = np.array([(6), (6, 6), (6, 6, 6), (6, 6, 6, 6), (6, 6, 6, 6, 6)])
# num_hidden_layers_arr_ax = np.arange(1, 6, 1)
#
# learning_rate = 0
# momentum = 0
#
# clf = ann.setup_ann(features_train, labels_train, features_test, labels_test)
# #plot_learning_curve(clf, features_train, labels_train, 'Neural-Network', np.linspace(0.01, 1.0, 10), 8)
# plot_validation_curve(clf, features_train, labels_train, 'Neural-Network', 'hidden_layer_sizes',
#                       hidden_layer_size_arr, hidden_layer_size_arr_ax)
# plot_validation_curve(clf, features_train, labels_train, 'Neural-Network', 'hidden_layer_sizes',
#                       num_hidden_layers_arr, num_hidden_layers_arr_ax)
# ann.fit_ann(clf, features_train, labels_train)
# pred = ann.predict_ann(clf, features_test)
# ann_precision, ann_recall, ann_f1 = ann.get_performance_ann(clf, pred, features_test, labels_test)


print('##################### Boosting ######################')

print('##################### kNN ######################')

# n_neighbors_arr = np.arange(1, 100, 5)
# distance_metric_arr = np.array(['euclidean', 'manhattan', 'chebyshev'])
#
#
# clf = knn.setup_knn(features_train, labels_train, features_test, labels_test)
#
# # plot learning curve
# #plot_learning_curve(clf, features_train, labels_train, 'kNN', np.linspace(.1, 1., 10), 8)
#
# # plot validation curve
# # plot_validation_curve(clf, features_train, labels_train, 'kNN', 'n_neighbors', n_neighbors_arr)
# plot_validation_curve(clf, features_train, labels_train, 'kNN', 'metric', distance_metric_arr)
#
# # fit
#
# knn.fit_knn(clf, features_train, labels_train)
#
# # predict testing set
# pred = knn.predict_knn(clf, features_test)
#
# knn_precision, knn_recall, knn_f1 = knn.get_performance_knn(clf, pred, features_test, labels_test)
