from sklearn.model_selection import learning_curve, validation_curve
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler


from sklearn.neural_network import MLPClassifier


def plot_learning_curve(clf, features_train, labels_train, clf_name, dataset_name, train_sizes=np.linspace(0.01, 1.0, 5),
                        cv=5, shuffle=True):
    """

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
                                                                      train_sizes=train_sizes, cv=cv, shuffle=shuffle)

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
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves for '+ clf_name + ' classifier on '+ dataset_name, fontsize=14)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.savefig('images/learning-curve-' + clf_name + '.png')
    plt.close()
    return plt


def plot_validation_curve(clf, features_train, labels_train, clf_name, dataset_name, param_name, param_range, x_arr=None, cv=5):
    """

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


    train_scores, validation_scores = validation_curve(clf, features_train, labels_train, param_name, param_range)

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
    plt.ylabel('Score', fontsize=14)
    plt.xlabel(param_name, fontsize=14)
    plt.title(dataset_name + '- ' + 'Validation Curve for ' + clf_name + ' classifier, parameter: ' + param_name, fontsize=14)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.savefig('images/' + dataset_name + 'validation-curve-' + clf_name + '-' + param_name+ str(datetime.now()) +'.png')
    plt.close()
    return plt

    return plt
