import mlrose
import pandas as pd
import traitlets.utils.bunch
from mlrose import NNGSRunner
from sklearn.datasets import load_iris
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from plotting import plot_learning_curve


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


abalone_ternary = load_iris()#process_abalone_ternary()
# ds_name = 'Abalone Ternary'
#
#
features_train, features_test, labels_train, labels_test = train_test_split(abalone_ternary.data,
                                                                            abalone_ternary.target, test_size=0.2,
                                                                            random_state=123)
#
# # scaling used https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
scaler = StandardScaler()
scaler.fit(features_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

# https://mlrose.readthedocs.io/en/stable/source/tutorial3.html
one_hot = OneHotEncoder()

labels_train_hot = one_hot.fit_transform(labels_train.reshape(-1, 1)).todense()
labels_test_hot = one_hot.transform(labels_test.reshape(-1, 1)).todense()


############################################################
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
#
# data = load_iris()
# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(abalo.data, data.target,
#                                                     test_size = 0.2, random_state = 3)
#
# # Normalize feature data
# scaler = MinMaxScaler()
#
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # # One hot encode target values
# one_hot = OneHotEncoder()
#
# y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).toarray()
# y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).toarray()
############################################################

def run_nn_experiment(runner, output_directory, experiment_name, experiment_parameters, seed, grid_search_parameters,
                      **kwargs):
    all_args = {**experiment_parameters, **kwargs, }
    results = runner(seed=seed,
                     experiment_name=experiment_name,
                     grid_search_parameters=grid_search_parameters,
                     output_directory=output_directory,
                     **all_args).run()

    print(results)

    return results

#
experiment_parameters = {
    'x_train': features_train,
    'y_train': labels_train_hot,
    'x_test': features_test,
    'y_test': labels_test_hot,
    'max_attempts': 100,
    'early_stopping': True
}

sa_hp_params = {
    'temperature': [10],
    'max_iters': [2**i for i in range(8, 15)],
    'learning_rate_init': [0.001],
    'hidden_layer_sizes': [[3, 3]],
    'activation': [mlrose.neural.activation.relu],
}

run_nn_experiment(runner=NNGSRunner,
                  algorithm=mlrose.algorithms.ga.genetic_alg,
                  output_directory='out/sa',
                  experiment_name='iters',
                  experiment_parameters=experiment_parameters,
                  grid_search_parameters=sa_hp_params,
                  iteration_list= 2 ** np.arange(8, 15),
                  seed=90)


# nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
#                                  algorithm = 'gradient_descent', max_iters = 1000, \
#                                  bias = True, is_classifier = True, learning_rate = 0.0001, \
#                                  early_stopping = True, clip_max = 5, max_attempts = 100, \
#                                  random_state = 3)
# #
# nn_model1.fit(features_train, labels_train_hot)
# y_test_pred = nn_model1.predict(features_test)
#
# y_test_accuracy = accuracy_score(labels_test_hot, y_test_pred)
#
# print(y_test_accuracy)

# rhc_hp_params = {
#     'restart': [1, 5, 10],
#     'max_iters': [1000],  # [2**i for i in range(8, 16)],
#     'learning_rate_init': [0.001],
#     'hidden_layer_sizes': [[5, 5, 5, 5]],
#     'activation': [mlrose.neural.activation.relu],
# }
#
# run_nn_experiment(runner=NNGSRunner,
#                   algorithm=mlrose.algorithms.rhc.random_hill_climb,
#                   output_directory='out/rhc',
#                   experiment_name='tuning',
#                   experiment_parameters=experiment_parameters,
#                   grid_search_parameters=rhc_hp_params,
#                   iteration_list=[1000],  # 2 ** np.arange(8, 16),
#                   seed=90)
#
# ga_hp_params = {
#     'pop_size': [200, 500, 1000],
#     'max_iters': [1000],  # [2**i for i in range(8, 16)],
#     'learning_rate_init': [0.001],
#     'hidden_layer_sizes': [[5, 5, 5, 5]],
#     'activation': [mlrose.neural.activation.relu],
# }
#
# run_nn_experiment(runner=NNGSRunner,
#                   algorithm=mlrose.algorithms.ga.genetic_alg,
#                   output_directory='out/ga',
#                   experiment_name='tuning',
#                   experiment_parameters=experiment_parameters,
#                   grid_search_parameters=ga_hp_params,
#                   iteration_list=[1000],  # 2 ** np.arange(8, 16),
#                   seed=90)

#
# # print('##################### Neural Network ######################')
# scoring = {'f1': make_scorer(f1_score, average='macro'), 'balanced_accuracy': 'balanced_accuracy'}
# refit = 'balanced_accuracy'
#
# clf_name = 'Neural Network'
# clf = MLPClassifier()
# clf = ann.setup_ann(features_train, labels_train)
#
# # find optimal parameters
# activation_arr = np.array(['identity', 'logistic', 'tanh', 'relu'])
# solver_arr = np.array(['lbfgs', 'sgd', 'adam'])
#
# # parameters = {'activation': activation_arr, 'solver': solver_arr,
# #               'hidden_layer_sizes': [(10, 10), (50, 50, 50), (100,)]}
# # ---------Grid Search Results----------
# #    Dataset: Wine Quality
# #    Learner: Neural Network
# # Best Paramters: {'activation': 'tanh', 'hidden_layer_sizes': (50, 50, 50), 'solver': 'lbfgs'}
# #
# # --------------------------------------------
#
# # parameters:
# #   1. the size of hidden layers
# hidden_layer_size_arr_ax = np.arange(50, 200, 50)
# hidden_layer_size_arr = np.array([(i, i, i, i, i) for i in hidden_layer_size_arr_ax])
#
# #   2. number of layers
# num_hidden_layers_arr = []
# for i in range(4, 8):
#     temp = (200,) * i
#     num_hidden_layers_arr.append(temp)
# num_hidden_layers_arr = np.array(num_hidden_layers_arr)
# num_hidden_layers_arr_ax = np.arange(4, 8, 1)
#
# #   3. alpha
# alpha_arr = np.arange(.00001, .01, .0005)
#
# # grid search
# # parameters = {'hidden_layer_sizes' : [(200), (200, 200), (200, 200, 200), (200, 200, 200, 200)]}
# # gc_clf = GridSearchCV(clf, parameters, cv=5, scoring=scoring, refit= refit)
# # gc_clf.fit(features_train, labels_train)
# # gc_best_params = gc_clf.best_params_
# # write_gridsearch_results(ds_name, clf_name, gc_best_params)
# #
#
# # set up a tuned clf
# # since grid search takes forever:
# tuned_clf = ann.setup_ann(features_train, labels_train, {'hidden_layer_sizes': (300, 300, 300, 300)})
# # for grid search use: tuned_clf = ann.setup_ann(features_train, labels_train, gc_best_params)
#
# # plot learning curve
#
# plot_learning_curve(tuned_clf, features_train, labels_train, clf_name, ds_name, np.linspace(.01, 1., 10), 8)
#
#
#
# # fit
#
# ann.fit_ann(tuned_clf, features_train, labels_train)
#
# # predict testing set
# pred = ann.predict_ann(tuned_clf, features_test)
#
# ann_score, ann_precision, ann_recall, ann_f1 = ann.get_performance_ann(tuned_clf, pred, features_test, labels_test)
#
# write_performance(ds_name, clf_name, ann_score, ann_precision, ann_recall, ann_f1)
#
# # print('##################### Boosting ######################')
