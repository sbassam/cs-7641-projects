from sklearn.tree import DecisionTreeClassifier

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
from sklearn.model_selection import GridSearchCV

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
ds_name = 'Wine Quality'


# ds_name = 'abalone_ternary'

def write_performance(dataset_name, learner, score, precision, recall, f1):
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


def write_gridsearch_results(dataset_name, learner, best_params):
    """

    :param dataset_name: string, name of the dataset
    :param learner: string, name of the learner
    :param best_params: dict, result of the grid search
    :return:
    """
    f = open("gridsearch_results.txt", "a")
    statement = "\n----------Grid Search Results----------\n" \
                "   Dataset: %s\n" \
                "   Learner: %s\n" \
                "Best Paramters: %s\n" \
                "\n--------------------------------------------\n"
    f.write(statement % (dataset_name, learner, best_params))
    f.close()

    return


features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.3)

# scaling used https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
scaler = StandardScaler()
scaler.fit(features_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

###########baseline stats##############
# determine the baseline performance (out of the box performance)
# initialize
dt_baseline = dt.setup_dt(features_train, labels_train)
ann_baseline = ann.setup_ann(features_train, labels_train)
# bst_baseline = bst.setup_bst(features_train, labels_train)
# svm_baseline = svm.setup_svm(features_train, labels_train)
#knn_baseline = knn.setup_knn(features_train, labels_train)
# fit
dt.fit_dt(dt_baseline, features_train, labels_train)
ann.fit_ann(ann_baseline, features_train, labels_train)
# bst.fit_bst(bst_basline, features_train, labels_train)
# svm.fit_svm(svm_basline, features_train, labels_train)
#knn.fit_knn(knn_baseline, features_train, labels_train)
# predict
bl_dt_pred = dt.predict_dt(dt_baseline, features_test)
bl_ann_pred = ann.predict_ann(ann_baseline, features_test)
# bl_bst_pred = bst.predict_bst(bst_baseline, features_test)
# bl_svm_pred = svm.predict_svm(svm_baseline, features_test)
# bl_knn_pred = knn.predict_knn(knn_baseline, features_test)

# performance
bl_dt_score, bl_dt_precision, bl_dt_recall, bl_dt_f1 = dt.get_performance_dt(dt_baseline, bl_dt_pred, features_test,
                                                                             labels_test)
bl_ann_score, bl_ann_precision, bl_ann_recall, bl_ann_f1 = ann.get_performance_ann(ann_baseline, bl_ann_pred,
                                                                                   features_test, labels_test)
# bl_bst_score, bl_bst_precision, bl_bst_recall, bl_bst_f1 = bst.get_performance_bst(bst_baseline, bl_bst_pred,
#                                                                                    features_test, labels_test)
# bl_svm_score, bl_svm_precision, bl_svm_recall, bl_svm_f1 = svm.get_performance_svm(svm_baseline, bl_svm_pred,
#                                                                                    features_test, labels_test)
#bl_knn_score, bl_knn_precision, bl_knn_recall, bl_knn_f1 = knn.get_performance_knn(knn_baseline, bl_knn_pred,
#                                                                                   features_test, labels_test)
# write
write_performance(ds_name, 'Base Line Decision Tree', bl_dt_score, bl_dt_precision, bl_dt_recall, bl_dt_f1)
write_performance(ds_name, 'Base Line Neural Network', bl_ann_score, bl_ann_precision, bl_ann_recall, bl_ann_f1)
# write_performance(ds_name, 'Base Line Boosting', bl_bst_score, bl_bst_precision, bl_bst_recall, bl_bst_f1)
# write_performance(ds_name, 'Base Line SVM', bl_svm_score, bl_svm_precision, bl_svm_recall, bl_svm_f1)
#write_performance(ds_name, 'Base Line kNN', bl_knn_score, bl_knn_precision, bl_knn_recall, bl_knn_f1)
#################################################

#
# print('##################### Decision tree ######################')
#
clf_name = 'Decision Tree'

#
#
clf = dt.setup_dt(features_train, labels_train)

#

# find optimal parameters
parameters = {'max_depth': np.arange(1, 20, 2), 'min_samples_leaf': np.arange(2, 50, 2)}
gc_clf = GridSearchCV(clf, parameters, cv=5)
gc_clf.fit(features_train, labels_train)
gc_best_params = gc_clf.best_params_
write_gridsearch_results(ds_name, clf_name, gc_best_params)
#

# set up a tuned clf
tuned_clf = dt.setup_dt(features_train, labels_train, gc_best_params)

# plot learning curve

plot_learning_curve(tuned_clf, features_train, labels_train, clf_name, ds_name, np.linspace(.01, 1., 10), 8)

# plot validation curve

# n_neighbors_arr = np.arange(1, 100, 5)
max_depth_arr = np.arange(1, 100, 5)
max_leaf_nodes_arr = np.arange(2, 100, 5)
min_samples_leaf_arr = np.arange(1, 100, 5)
criterion_arr = np.array(['gini', 'entropy'])

plot_validation_curve(clf, features_train, labels_train, clf_name, ds_name, 'max_depth', max_depth_arr)
plot_validation_curve(clf, features_train, labels_train, clf_name, ds_name, 'min_samples_leaf',
                      min_samples_leaf_arr)

# fit

dt.fit_dt(tuned_clf, features_train, labels_train)

# predict testing set
pred = dt.predict_dt(tuned_clf, features_test)

dt_score, dt_precision, dt_recall, dt_f1 = dt.get_performance_dt(tuned_clf, pred, features_test, labels_test)

write_performance(ds_name, clf_name, dt_score, dt_precision, dt_recall, dt_f1)





print('##################### Neural Network ######################')

clf_name = 'Neural Network'
clf = ann.setup_ann(features_train, labels_train)

# find optimal parameters
activation_arr = np.array(['identity', 'logistic', 'tanh', 'relu'])
solver = np.array(['lbfgs', 'sgd', 'adam'])
hidden_layer_size_arr_ax = np.arange(32, 52, 4)
hidden_layer_size_arr = np.array([(i, i, i) for i in hidden_layer_size_arr_ax])

num_hidden_layers_arr = np.array([(40, 40), (40, 40, 40), (40, 40, 40, 40), (40, 40, 40, 40, 40), (40, 40, 40, 40, 40, 40)])
num_hidden_layers_arr_ax = np.arange(2, 7, 1)

parameters = {'activation': activation_arr}
gc_clf = GridSearchCV(clf, parameters, cv=5)
gc_clf.fit(features_train, labels_train)
gc_best_params = gc_clf.best_params_
write_gridsearch_results(ds_name, clf_name, gc_best_params)
#

# set up a tuned clf
tuned_clf = ann.setup_ann(features_train, labels_train, gc_best_params)

# plot learning curve

plot_learning_curve(tuned_clf, features_train, labels_train, clf_name, ds_name, np.linspace(.01, 1., 10), 8)

# plot validation curve


learning_rate = 0
momentum = 0

if ds_name == 'Wine Quality':
    hidden_layer_size_arr_ax = np.arange(1, 20, 4)
    hidden_layer_size_arr = np.array([(i, i, i) for i in hidden_layer_size_arr_ax])

    num_hidden_layers_arr = np.array([(6), (6, 6), (6, 6, 6), (6, 6, 6, 6), (6, 6, 6, 6, 6)])
    num_hidden_layers_arr_ax = np.arange(1, 6, 1)

    learning_rate = 0
    momentum = 0

# # set up a classifier for the validation curve
# clf = ann.setup_ann(features_train, labels_train)

plot_validation_curve(tuned_clf, features_train, labels_train, clf_name, ds_name, 'hidden_layer_sizes',
                      hidden_layer_size_arr)
plot_validation_curve(tuned_clf, features_train, labels_train, clf_name, ds_name, 'num_hidden_layers',
                      num_hidden_layers_arr)


# fit

ann.fit_ann(tuned_clf, features_train, labels_train)

# predict testing set
pred = ann.predict_ann(tuned_clf, features_test)

ann_score, ann_precision, ann_recall, ann_f1 = ann.get_performance_ann(tuned_clf, pred, features_test, labels_test)

write_performance(ds_name, clf_name, ann_score, ann_precision, ann_recall, ann_f1)


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

print('##################### kNN ######################')
clf_name = 'kNN'

n_neighbors_arr = np.arange(1, 50, 5)
distance_metric_arr = np.array(['euclidean', 'manhattan', 'chebyshev'])

# set up a simple knn with no hyper paramaters
clf = knn.setup_knn(features_train, labels_train)

# find optimal parameters
parameters = {'n_neighbors': np.arange(1, 100, 5), 'metric': distance_metric_arr}
gc_clf = GridSearchCV(clf, parameters, cv=5)
gc_clf.fit(features_train, labels_train)
gc_best_params = gc_clf.best_params_
write_gridsearch_results(ds_name, clf_name, gc_best_params)
#

# set up a tuned clf
tuned_clf = knn.setup_knn(features_train, labels_train, gc_best_params)

# plot learning curve
plot_learning_curve(tuned_clf, features_train, labels_train, clf_name, ds_name, np.linspace(.1, 1., 10), 8)

# plot validation curve

plot_validation_curve(clf, features_train, labels_train, 'kNN', ds_name, 'n_neighbors', n_neighbors_arr)
plot_validation_curve(clf, features_train, labels_train, 'kNN', ds_name, 'metric', distance_metric_arr)

# fit

knn.fit_knn(tuned_clf, features_train, labels_train)

# predict testing set
pred = knn.predict_knn(tuned_clf, features_test)

knn_score, knn_precision, knn_recall, knn_f1 = knn.get_performance_knn(tuned_clf, pred, features_test, labels_test)

write_performance(ds_name, clf_name, knn_score, knn_precision, knn_recall, knn_f1)
