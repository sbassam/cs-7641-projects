from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import time
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import make_scorer, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
import loader
from yellowbrick.cluster import KElbowVisualizer

#ds_name = 'abalone-ternary'
from dr import run_pca, run_ica
from clustering import run_gm, run_km
from gs import run_gs
from plotting import plot_elbow, plot_pca, plot_ica, plot_ica_avg_kurtosis

ds_name = 'wine-quality'
ds_name = 'abalone-ternary'
#ds_name = 'iris'
run_grid = False
run_grid_p4 = False
run_grid_p5 = False

if ds_name == 'abalone-ternary':
    # get the data
    data = loader.process_abalone_ternary()
elif ds_name == 'wine-quality':
    # get the data
    data = loader.process_wine_quality()
elif ds_name == 'iris':
    data = load_iris()
else:
    print('no dataset with that name')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=3)
# Scale
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def write_performance(dataset_name, learner, score, f1):
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
    f = open("out/performance_results.txt", "a")

    statement = "\n----------Performance Summary----------\n" \
                "   Dataset: %s\n" \
                "   Learner: %s\n" \
                "Balanced Accuracy Score: %s\n" \
                "F1: %s\n" \
                "\n--------------------------------------------\n"
    f.write(statement % (dataset_name, learner, score, f1))
    f.close()

    return


# plot elbow
# n_clusters = plot_elbow(ds_name, 'KMeans', X_train)
# TODO: SILHOUTTE METHOD
# plot silhoutte find the best k

# TODO: run Gaussian Mixture
# run_gm(X_train, n_clusters)


# Dimensionality Reduction
# PCA
X_train_pca = X_train.copy()
pca, X_train_pca = run_pca(X_train_pca)
# plot_pca(ds_name, pca)
#
# ICA
X_train_ica = X_train.copy()
# X_test_ica will be used in the last part
X_test_ica = X_test.copy()
ica, X_train_ica, X_test_ica, kurtosis_list = run_ica(X_train_ica, X_test_ica)
plot_ica_avg_kurtosis(ds_name, kurtosis_list)
plot_ica(ds_name, ica)

#n_clusters = plot_elbow(ds_name, 'KMeans', X_train_pca)
#n_clusters = plot_elbow(ds_name, 'KMeans', X_train_ica)

# NN baseline from assignment 1
if run_grid:
    clf = run_gs(X_train, y_train, ds_name)
else:
    clf = MLPClassifier(random_state=69, hidden_layer_sizes=(100, 100, 100), alpha=0.01)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
write_performance(ds_name, 'ann', balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average='micro'))

# # NN + dimensionality reduction
# # NN + pca
# X_test_pca = X_test.copy()
# pca, X_test_pca = run_pca(X_test_pca)
#
#
# if run_grid_p4:
#     clf = run_gs(X_train_pca, y_train, ds_name, dr_type='pca')
# else:
#     clf = MLPClassifier(random_state=69, hidden_layer_sizes=(250, 250, 250), alpha=0.001)
# clf.fit(X_train_pca, y_train)
# pred = clf.predict(X_test_pca)
# write_performance(ds_name, 'ann-pca', balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average='micro'))

# NN + ica

if run_grid_p4:
    clf = run_gs(X_train_ica, y_train, ds_name, dr_type='ica')
else:
    clf = MLPClassifier(random_state=69, hidden_layer_sizes=(250, 250, 250), alpha=0.01)
clf.fit(X_train_ica, y_train)
pred = clf.predict(X_test_ica)
write_performance(ds_name, 'ann-ica', balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average='micro'))

# TODO: NN + rp
# if run_grid_p4:
#     clf = run_gs(X_train_rp, y_train, ds_name, dr_type='rp')
# else:
#     clf = MLPClassifier(random_state=69, hidden_layer_sizes=(100, 100, 100), alpha=0.01)
# clf.fit(X_train_rp, y_train)
# pred = clf.predict(X_test)
# write_performance(ds_name, 'ann-rp', balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average='micro'))

# TODO: NN + ig
# if run_grid_p4:
#     clf = run_gs(X_train_ig, y_train, ds_name, dr_type='ig')
# else:
#     clf = MLPClassifier(random_state=69, hidden_layer_sizes=(100, 100, 100), alpha=0.01)
# clf.fit(X_train_ig, y_train)
# pred = clf.predict(X_test)
# write_performance(ds_name, 'ann-ig', balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average='micro'))


# # # NN + dimensionality reduction + clustering
#  TODO: kmeans
n_clusters = 5
run_km(X_train, y_train, n_clusters)
# # TODO: expectation maximization
# run_gm(X_train_pca)












