from sklearn.model_selection import learning_curve, validation_curve, train_test_split
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV
import pandas as pd


def plot_learning_curve(clf, features_train, labels_train, clf_name, dataset_name,
                        train_sizes=np.linspace(0.01, 1.0, 5), cv=5, shuffle=True, scoring='balanced_accuracy'):
    """

    :param scoring:
    :param clf: estimator
    :param features_train: training set
    :param labels_train: labels for training set
    :param clf_name: string
    :param dataset_name: string the name of the dataset
    :param train_sizes: array of set sizes.
    :param cv: cross validation folds
    :param shuffle: whether or not to shuffle for the cross validation operation
    :return: the plt object

    uses:
    https://www.dataquest.io/blog/learning-curves-machine-learning/
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    """

    train_sizes_abs, train_scores, validation_scores = learning_curve(clf, features_train, labels_train,
                                                                      train_sizes=train_sizes, cv=cv,
                                                                      shuffle=shuffle, scoring=scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    plt.style.use('seaborn')

    plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes_abs, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes_abs, validation_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # plt.plot(train_sizes_abs, train_scores.mean(axis = 1), label = 'Training error')
    # plt.plot(train_sizes_abs, validation_scores.mean(axis = 1), label = 'Validation error')
    plt.ylabel('Balanced Accuracy Score', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves for ' + clf_name + ' classifier on ' + dataset_name, fontsize=14)
    plt.legend()
    plt.savefig('images/' + dataset_name + '-learning-curve-' + clf_name + '-' + str(datetime.now()) + '.png')
    plt.close()
    return plt


def plot_validation_curve(clf, features_train, labels_train, clf_name, dataset_name, param_name, param_range,
                          x_arr=None, cv=5, scoring='balanced_accuracy'):
    """

    :param scoring:
    :param scoring:
    :param dataset_name: string name of the dataset
    :param clf_name: string name of the classifier
    :param x_arr: array of x coordinate
    :param clf: estimator
    :param features_train: training vector
    :param labels_train: target vector
    :param param_name: the hyper-parameter to vary
    :param param_range: the range for the paramater variation
    :param cv: cross validation number of folds
    :return: plot object

    source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
    """

    if x_arr is None:
        x_arr = param_range

    train_scores, validation_scores = validation_curve(clf, features_train, labels_train, param_name, param_range,
                                                       cv=cv, scoring=scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    plt.style.use('seaborn')

    plt.fill_between(x_arr, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(x_arr, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    plt.plot(x_arr, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(x_arr, validation_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # plt.plot(train_sizes_abs, train_scores.mean(axis = 1), label = 'Training error')
    # plt.plot(train_sizes_abs, validation_scores.mean(axis = 1), label = 'Validation error')
    plt.ylabel('Balanced Accuracy Score', fontsize=14)
    plt.xlabel(param_name, fontsize=14)
    plt.title(dataset_name + '- ' + 'Validation Curve for ' + clf_name + ' classifier, parameter: ' + param_name,
              fontsize=14)
    plt.legend()
    plt.savefig(
        'images/' + dataset_name + 'validation-curve-' + clf_name + '-' + param_name + str(datetime.now()) + '.png')
    plt.close()
    return plt


def plot_time_vs_iterations(features_train, labels_train, clf, clf_name, dataset_name, gc_params):
    x_arr = []
    iter_data = pd.DataFrame()
    for size in np.arange(.01, 1., .05):
        iter_train, iter_test, iter_y_train, iter_y_test = train_test_split(features_train, labels_train,
                                                                            train_size=size)

        # gc_params = {'max_iter': np.array([5000])}  # {'max_iter': np.arange(200, 800, 100) }
        gc_clf = GridSearchCV(clf, gc_params, cv=5, scoring='balanced_accuracy')
        gc_clf.fit(iter_train, iter_y_train)
        x_arr.append(iter_train.shape[0])
        #     fit_time_list[iter_train.shape[0]]=gc_clf.cv_results_['mean_fit_time'][0]
        #     accuracy_list[iter_train.shape[0]]=gc_clf.cv_results_['mean_test_score'][0]
        iter_data = pd.concat([iter_data, pd.DataFrame(gc_clf.cv_results_)])

    plt.style.use('seaborn')

    plt.fill_between(x_arr, iter_data.mean_fit_time - iter_data.std_fit_time,
                     iter_data.mean_fit_time + iter_data.std_fit_time, alpha=0.1,
                     color="r")

    plt.plot(x_arr, iter_data.mean_fit_time, 'o-', color="r",
             label="Iteration time")

    plt.ylabel('Mean Fit time (sec)', fontsize=14)
    plt.xlabel('Training Size', fontsize=14)
    plt.title(dataset_name + '- ' + 'Validation Curve for ' + clf_name + ' classifier, parameter: ',
              fontsize=14)
    plt.legend()
    plt.savefig(
        'images/' + dataset_name + 'time-vs-iteration-' + clf_name + '-'  + str(datetime.now()) + '.png')
    plt.close()
    return plt

def plot_accuracy_vs_iterations(features_train, labels_train, clf, clf_name, dataset_name, gc_params):
    x_arr = []
    iter_data = pd.DataFrame()
    for size in np.arange(.01, 1., .05):
        iter_train, iter_test, iter_y_train, iter_y_test = train_test_split(features_train, labels_train,
                                                                            train_size=size)

        # gc_params = {'max_iter': np.array([5000])}  # {'max_iter': np.arange(200, 800, 100) }
        gc_clf = GridSearchCV(clf, gc_params, cv=5, scoring='balanced_accuracy')
        gc_clf.fit(iter_train, iter_y_train)
        x_arr.append(iter_train.shape[0])
        #     fit_time_list[iter_train.shape[0]]=gc_clf.cv_results_['mean_fit_time'][0]
        #     accuracy_list[iter_train.shape[0]]=gc_clf.cv_results_['mean_test_score'][0]
        iter_data = pd.concat([iter_data, pd.DataFrame(gc_clf.cv_results_)])

    plt.style.use('seaborn')

    plt.fill_between(x_arr, iter_data.mean_test_score - iter_data.std_test_score,
                     iter_data.mean_test_score + iter_data.std_test_score, alpha=0.1,
                     color="r")

    plt.plot(x_arr, iter_data.mean_test_score, 'o-', color="r")

    plt.ylabel('Balanced Accuracy Score', fontsize=14)
    plt.xlabel('Training Size', fontsize=14)
    plt.title(dataset_name + '- ' + 'Validation Curve for ' + clf_name + ' classifier, parameter: ',
              fontsize=14)
    plt.legend()
    plt.savefig(
        'images/' + dataset_name + 'accuracy-vs-iteration-' + clf_name + '-'  + str(datetime.now()) + '.png')
    plt.close()
    return plt


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
