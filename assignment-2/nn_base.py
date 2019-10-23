import time

import mlrose
import pandas as pd
import traitlets.utils.bunch
from mlrose import NNGSRunner
from sklearn.datasets import load_iris
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import numpy as np
from plotting import plot_nn_score_vs_iters, plot_nn_fit_time_vs_iter, plot_nn_loss_vs_iter


def process_abalone_ternary():
    df = pd.read_csv('data/abalone.data', names=["Sex", "Length", "Diameter", "Height",
                                                 "Whole weight", "Shucked weight", "Viscera weight",
                                                 "Shell weight", "Rings"])
    df = df[(df["Height"] != 1.13) & (df['Height'] != 0.515)]

    # deal with categorical data
    df.loc[df.Sex == 'M', 'Male'] = 1.
    df.loc[df.Sex == 'F', 'Female'] = 1.
    df.loc[df.Sex == 'I', 'Infant'] = 1.
    df.fillna(0, inplace=True)

    # bucketize rings
    df.loc[df.Rings < 11, 'Rings'] = 1.
    df.loc[(df.Rings < 21) & (df.Rings > 10), 'Rings'] = 2.
    df.loc[df.Rings > 20, 'Rings'] = 3.

    return traitlets.Bunch(
        data=df[['Male', 'Female', 'Infant', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                 'Viscera weight', 'Shell weight']].values,
        target=df[['Rings']].values,
        target_names=df["Rings"].unique(),
        DESCR='abalone dataset...',
        feature_names=['Male', 'Female', 'Infant', "Length", "Diameter", "Height",
                       "Whole weight", "Shucked weight", "Viscera weight",
                       "Shell weight"],
    )


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
                "Balanced Accuracy Score: %s\n" \
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
                "Best Parameters: %s\n" \
                "\n--------------------------------------------\n"
    f.write(statement % (dataset_name, learner, best_params))
    f.close()

    return


abalone_ternary = process_abalone_ternary()
ds_name = 'Abalone Ternary'
features_train, features_test, labels_train, labels_test = train_test_split(abalone_ternary.data,
                                                                            abalone_ternary.target, test_size=0.2,
                                                                            random_state=3)
# scaling used https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
scaler = StandardScaler()
scaler.fit(features_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

# https://mlrose.readthedocs.io/en/stable/source/tutorial3.html
one_hot = OneHotEncoder()

labels_train_hot = one_hot.fit_transform(labels_train.reshape(-1, 1)).todense()
labels_test_hot = one_hot.transform(labels_test.reshape(-1, 1)).todense()


# def run_nn_gc(alg_name, features_train, labels_train_hot, features_test, labels_test_hot):
#     x_arr = [1, 50, 100, 200, 400, 800]
#     iter_data = pd.DataFrame()
#     csv_path = 'out/nn_' + str(alg_name) + '.csv'
#
#     for i in range(len(x_arr)):
#         start = time.time()
#
#         if alg_name == 'random_hill_climb':
#             clf = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu', algorithm=alg_name,
#                                        bias=True, is_classifier=True, learning_rate=0.001,
#                                        early_stopping=True, clip_max=5, max_attempts=100, restarts=1
#                                        )
#         if alg_name == 'simulated_annealing':
#             clf = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu',
#                                        algorithm=alg_name,
#                                        bias=True, is_classifier=True, learning_rate=0.001,
#                                        early_stopping=True, clip_max=5, max_attempts=100,
#                                        schedule=mlrose.GeomDecay(init_temp=1))
#         if alg_name == 'genetic_alg':
#             clf = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu',
#                                        algorithm=alg_name,
#                                        bias=True, is_classifier=True, learning_rate=0.001,
#                                        early_stopping=True, clip_max=5, max_attempts=100, pop_size=500)
#         if alg_name == 'gradient_descent':
#             clf = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu',
#                                        algorithm=alg_name,
#                                        bias=True, is_classifier=True, learning_rate=0.001,
#                                        early_stopping=True, clip_max=5, max_attempts=100)
#
#         parameters = {'max_iters': [x_arr[i]]}
#         gc_clf = GridSearchCV(clf, parameters, cv=3, scoring='accuracy', return_train_score=True)
#         gc_clf.fit(features_train, labels_train_hot)
#         finish = time.time()
#         df = gc_clf.cv_results_
#         df['loss'] = gc_clf.best_estimator_.loss
#         print('max iter: ', x_arr[i], '\n', 'algorithm: ', alg_name, '\n', 'time: ', finish - start)
#         iter_data = pd.concat([iter_data, pd.DataFrame(df)])
#     iter_data.to_csv(csv_path, index=None)
#
#     # use testing set on the last iteration (1600)
#     y_test_pred = gc_clf.predict(features_test)
#
#     y_test_accuracy = accuracy_score(labels_test_hot, y_test_pred)
#     y_test_f1 = f1_score(labels_test_hot, y_test_pred, average='macro')
#
#     print('accuracy: ', y_test_accuracy)
#     print('f1: ', y_test_f1)
#     return

def run_nn(alg_name):
    # start pre processing
    data = process_abalone_ternary()
    # snippet from https://mlrose.readthedocs.io/en/stable/source/tutorial3.html
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, \
                                                        test_size=0.2, random_state=3)

    # Normalize feature data
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One hot encode target values
    one_hot = OneHotEncoder()

    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

    csv_path = 'out/nn_' + str(alg_name) + '.csv'
    x_arr = [1, 50, 100, 200, 400, 800, 1600, 3200]
    cols = ['iter', 'train_accuracy', 'test_accuracy', 'test_f1', 'fit_time', 'loss']
    data = []
    for i in range(len(x_arr)):
        # TODO: for the life of me I CAN'T FIGURE OUT WHY THIS COMMENTED PART KEPT GIVING 0.65209987 ACCURACY!!!!
        # TODO: FIGURE OUT HOW COMMENTED IS DIFFERENT FROM BELOW IT.
        # start = time.time()
        # clf = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu', algorithm=alg_name,
        #                            bias=True, is_classifier=True, learning_rate=0.001,
        #                            early_stopping=True, clip_max=5, max_attempts=100,
        #                            restarts=1, schedule=mlrose.GeomDecay(init_temp=1),
        #                            pop_size=250, max_iters=x_arr[i], random_state=i)
        # clf.fit(features_train, labels_train_hot)
        # finish = time.time()
        # pred_train = clf.predict(features_train)
        # train_accuracy = accuracy_score(labels_train_hot, pred_train)
        # pred_test = clf.predict(features_test)
        # test_accuracy = accuracy_score(labels_test_hot, pred_test)
        # y_test_f1 = f1_score(labels_test_hot, pred_test, average='macro')
        # fit_time = finish - start
        # row = [x_arr[i], train_accuracy, test_accuracy, y_test_f1, fit_time, clf.loss]

        start = time.time()
        # https://mlrose.readthedocs.io/en/stable/source/tutorial3.html
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[10], activation='relu', \
                                         algorithm=alg_name, max_iters=400, \
                                         bias=True, is_classifier=True, learning_rate=0.001, \
                                         early_stopping=True, clip_max=5, max_attempts=100, \
                                         random_state=3)

        nn_model1.fit(X_train_scaled, y_train_hot)
        finish = time.time()
        fit_time = finish - start
        y_train_pred = nn_model1.predict(X_train_scaled)
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
        y_test_pred = nn_model1.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        y_test_f1 = f1_score(y_test_hot, y_test_pred, average='macro')
        row = [x_arr[i], y_train_accuracy, y_test_accuracy, y_test_f1, fit_time, nn_model1.loss]
        print(row)
        data.append(row)
        nn_model1 = []
    result = pd.DataFrame(data, columns=cols)
    result.to_csv(csv_path, index=None)
    return


alg_names = ['random_hill_climb', 'simulated_annealing', 'gradient_descent', 'genetic_alg']
for i in alg_names:
    run_nn(i)
    plot_nn_score_vs_iters(i)
    plot_nn_fit_time_vs_iter(i)
    plot_nn_loss_vs_iter(i)
