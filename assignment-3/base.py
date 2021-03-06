from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from datetime import datetime
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

# ds_name = 'abalone-ternary'
from dr import run_pca, run_ica, run_rp, run_svd
from clustering import run_gm, run_km
from gs import run_gs
from plotting import plot_elbow, plot_pca, plot_ica, plot_ica_avg_kurtosis, plot_sil_score, plot_BIC, \
    plot_rp_reconstruction_error, plot_svd_reconstruction_error

ds_name = 'wine-quality'
#ds_name = 'abalone-ternary'
# ds_name = 'iris'
run_grid = False
run_grid_p4 = False
run_grid_p5 = False
performance_stat_path = 'out/performance-stat' + str(datetime.now()) + '.csv'
performance_stat_col = ['ds_name', 'alg_name', 'n_clusters', 'v_measure_score', 'adjusted_rand_score', 'AMI_score',
                        'fit_time']
performance_stat_data = []
performance_stat_path_ann = 'out/performance-stat-ANN-' + str(datetime.now()) + '.csv'
performance_stat_col_ann = ['ds_name', 'alg_name', 'n_clusters', 'balanced_accuracy', 'f1','ANN-fit_time']
performance_stat_data_ann = []
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


###################  PART I.  ###################
print ('part 1')
# 1.1 K-Means
#       a) Find the best K using only training data
#           1) plot elbow
# n_clusters = plot_elbow(ds_name, 'KMeans', X_train)
#           2) plot Silhouette
best_k = plot_sil_score(X_train, ds_name, alg_name='Kmeans', kmax=10)
#       b) run KMeans and validate results
performance_stat, X_train_km = run_km(ds_name, X_train, y_train, best_k)
performance_stat_data.append(performance_stat)
# 1.2 Expectation Maximization
#       a) Find the best n_components
gm_best_cv_type, gm_best_n_components = plot_BIC(ds_name, X_train, max_n_components=15)
#       b) run GM and validate results
run_gm(ds_name, X_train, y_train, gm_best_n_components, cv_type=gm_best_cv_type)
#################################################

###################  PART II.  ###################
print ('part 2')

# 2.1 PCA
#       a) run PCA and transform training and testing data (test set will be used in part 4)
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()
pca, X_train_pca, X_test_pca = run_pca(X_train_pca, X_test_pca)
#       B) plot eigenvalues
plot_pca(ds_name, pca)
# 2.2 ICA
#       a) run ICA and transform training and testing data (test set will be used in part 4)
X_train_ica = X_train.copy()
X_test_ica = X_test.copy()
ica, X_train_ica, X_test_ica, kurtosis_means, kurtosis_stds = run_ica(X_train_ica, X_test_ica)
#       b) plot average kurtosis to show which n_component has the highest kurtosis on average
plot_ica_avg_kurtosis(ds_name, kurtosis_means, kurtosis_stds)
#       c) plot kurtosis for the best n_components.
plot_ica(ds_name, ica)
# 2.3 RP
#       a) run rp and transform training and testing data (test set will be used in part 4)
X_train_rp = X_train.copy()
X_test_rp = X_test.copy()
rp, X_train_rp, X_test_rp, rp_loss_means, rp_loss_stds = run_rp(X_train_rp, X_test_rp)
#       b) plot reconstruction error
plot_rp_reconstruction_error(ds_name, rp_loss_means, rp_loss_stds)

# 2.4 Truncated SVD
#       a) run SVD and transform training and testing data (test set will be used in part 4)
X_train_svd = X_train.copy()
X_test_svd = X_test.copy()
svd, X_train_svd, X_test_svd, svd_loss_means, svd_loss_stds = run_svd(X_train_svd, X_test_svd)
#       b) plot reconstruction error
plot_svd_reconstruction_error(ds_name, svd_loss_means, svd_loss_stds)
#################################################

###################  PART III.  ###################
print ('part 3')

# 3.1 K-Means
#   a) PCA
best_k = plot_sil_score(X_train_pca, ds_name, alg_name='KMeans-PCA', kmax=10)
performance_stat, X_train_km = run_km(ds_name, X_train_pca, y_train, best_k, alg_name='KMeans-PCA')
performance_stat_data.append(performance_stat)
#   b) ICA
best_k = plot_sil_score(X_train_ica, ds_name, alg_name='Kmeans-ICA', kmax=10)
performance_stat, X_train_km = run_km(ds_name, X_train_ica, y_train, best_k, alg_name='KMeans-ICA')
performance_stat_data.append(performance_stat)
#   c) RP
best_k = plot_sil_score(X_train_rp, ds_name, alg_name='Kmeans-RP', kmax=10)
performance_stat, X_train_km = run_km(ds_name, X_train_rp, y_train, best_k, alg_name='KMeans-RP')
performance_stat_data.append(performance_stat)
#   d) SVD
best_k = plot_sil_score(X_train_svd, ds_name, alg_name='Kmeans-SVD', kmax=10)
performance_stat, X_train_km = run_km(ds_name, X_train_svd, y_train, best_k, alg_name='KMeans-SVD')
performance_stat_data.append(performance_stat)
# 3.2 Expectation Maximization (GM)
#   a) PCA
gm_best_cv_type, gm_best_n_components = plot_BIC(ds_name, X_train_pca, max_n_components=15)
performance_stat, X_train_gm = run_gm(ds_name, X_train_pca, y_train, gm_best_n_components, cv_type=gm_best_cv_type,
                                      alg_name='GM-PCA')
performance_stat_data.append(performance_stat)
#   b) ICA
gm_best_cv_type, gm_best_n_components = plot_BIC(ds_name, X_train_ica, max_n_components=15)
performance_stat, X_train_gm = run_gm(ds_name, X_train_ica, y_train, gm_best_n_components, cv_type=gm_best_cv_type,
                                      alg_name='GM-ICA')
performance_stat_data.append(performance_stat)
#   c) RP
gm_best_cv_type, gm_best_n_components = plot_BIC(ds_name, X_train_rp, max_n_components=15)
performance_stat, X_train_gm = run_gm(ds_name, X_train_rp, y_train, gm_best_n_components, cv_type=gm_best_cv_type,
                                      alg_name='GM-RP')
performance_stat_data.append(performance_stat)
#   d) SVD
gm_best_cv_type, gm_best_n_components = plot_BIC(ds_name, X_train_svd, max_n_components=15)
performance_stat, X_train_gm = run_gm(ds_name, X_train_svd, y_train, gm_best_n_components, cv_type=gm_best_cv_type,
                                      alg_name='GM-SVD')
performance_stat_data.append(performance_stat)
#################################################

###################  PART IV.  ###################
print ('part 4')

# set ds_name for part 4 and 5
ds_name = 'abalone-ternary'
# NN baseline from assignment 1
if run_grid:
    clf = run_gs(X_train, y_train, ds_name)
else:
    clf = MLPClassifier(random_state=69, hidden_layer_sizes=(100, 100, 100), alpha=0.01)
start = time.time()
clf.fit(X_train, y_train)
end = time.time()
pred = clf.predict(X_test)
write_performance(ds_name, 'ann', balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average='micro'))
performance_stat_data_ann.append([ds_name, 'Baseline-ANN', X_train.shape[1], balanced_accuracy_score(y_test, pred),
                                  f1_score(y_test, pred, average='micro'), end-start])
# 4.1 PCA
if run_grid_p4:
    clf = run_gs(X_train_pca, y_train, ds_name, dr_type='pca')
else:
    clf = MLPClassifier(random_state=69, hidden_layer_sizes=(250, 250, 250), alpha=0.001)
start = time.time()
clf.fit(X_train_pca, y_train)
end = time.time()
pred = clf.predict(X_test_pca)
write_performance(ds_name, 'ann-pca', balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average='micro'))
performance_stat_data_ann.append([ds_name, 'PCA-ANN', X_train_pca.shape[1], balanced_accuracy_score(y_test, pred),
                                  f1_score(y_test, pred, average='micro'), end-start])
# 4.2 ICA
if run_grid_p4:
    clf = run_gs(X_train_ica, y_train, ds_name, dr_type='ica')
else:
    clf = MLPClassifier(random_state=69, hidden_layer_sizes=(250, 250, 250), alpha=0.01)
start = time.time()
clf.fit(X_train_ica, y_train)
end = time.time()
pred = clf.predict(X_test_ica)
write_performance(ds_name, 'ann-ica', balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average='micro'))
performance_stat_data_ann.append([ds_name, 'ICA-ANN', X_train_ica.shape[1], balanced_accuracy_score(y_test, pred),
                                  f1_score(y_test, pred, average='micro'), end-start])
# 4.3 RP
if run_grid_p4:
    clf = run_gs(X_train_rp, y_train, ds_name, dr_type='rp')
else:
    clf = MLPClassifier(random_state=69, hidden_layer_sizes=(100, 100, 100), alpha=0.01)
start = time.time()
clf.fit(X_train_rp, y_train)
end = time.time()
pred = clf.predict(X_test_rp)
write_performance(ds_name, 'ann-rp', balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average='micro'))
performance_stat_data_ann.append([ds_name, 'RP-ANN', X_train_rp.shape[1], balanced_accuracy_score(y_test, pred),
                                  f1_score(y_test, pred, average='micro'), end-start])

# 4.4 SVD
if run_grid_p4:
    clf = run_gs(X_train_svd, y_train, ds_name, dr_type='ig')
else:
    clf = MLPClassifier(random_state=69, hidden_layer_sizes=(100, 100, 100), alpha=0.01)
start = time.time()
clf.fit(X_train_svd, y_train)
end = time.time()
pred = clf.predict(X_test_svd)
write_performance(ds_name, 'ann-ig', balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average='micro'))
performance_stat_data_ann.append([ds_name, 'SVD-ANN', X_train_svd.shape[1], balanced_accuracy_score(y_test, pred),
                                  f1_score(y_test, pred, average='micro'), end-start])
#################################################

###################  PART V.  ###################
print ('part 5')

# 5.1 K-Means
#   a) run km
best_k = plot_sil_score(X_train, ds_name, alg_name='Kmeans-ANN', kmax=10)
performance_stat, X_train_km, X_test_km = run_km(ds_name, X_train, y_train, best_k, X_test, alg_name='KMeans-ANN')
performance_stat_data.append(performance_stat)
#   b) grid search on new training data
if run_grid_p5:
    clf = run_gs(X_train_km, y_train, ds_name, cluster_type='KMeans')
else:
    clf = MLPClassifier(random_state=69, hidden_layer_sizes=(250, 250, 250), alpha=0.01)
#   c) run NN and plot performance
start = time.time()
clf.fit(X_train_km, y_train)
end = time.time()
pred = clf.predict(X_test_km)
write_performance(ds_name, 'KMeans-ANN', balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average='micro'))
performance_stat_data_ann.append([ds_name, 'KMeans-ANN', best_k, balanced_accuracy_score(y_test, pred),
                                  f1_score(y_test, pred, average='micro'),end-start])
# 5.2 Expectation Maximization
#   a) run gm
gm_best_cv_type, gm_best_n_components = plot_BIC(ds_name, X_train, max_n_components=15)
performance_stat, X_train_gm, X_test_gm = run_gm(ds_name, X_train, y_train, gm_best_n_components,
                                                 cv_type=gm_best_cv_type, X_test=X_test, alg_name='GM-ANN')
performance_stat_data.append(performance_stat)
#   b) grid search on new training data
if run_grid_p5:
    clf = run_gs(X_train_gm, y_train, ds_name, cluster_type='GM')
else:
    clf = MLPClassifier(random_state=69, hidden_layer_sizes=(200, 200, 200), alpha=0.01)
#   c) run NN and plot performance
start = time.time()
clf.fit(X_train_gm, y_train)
end = time.time()
pred = clf.predict(X_test_gm)
write_performance(ds_name, 'GM-ANN', balanced_accuracy_score(y_test, pred), f1_score(y_test, pred, average='micro'))
performance_stat_data_ann.append([ds_name, 'GM-ANN', gm_best_n_components, balanced_accuracy_score(y_test, pred),
                                  f1_score(y_test, pred, average='micro'), end-start])

performance_stat_data = pd.DataFrame(performance_stat_data, columns=performance_stat_col)
performance_stat_data.to_csv(performance_stat_path, index=None)

performance_stat_data_ann = pd.DataFrame(performance_stat_data_ann, columns=performance_stat_col_ann)
performance_stat_data_ann.to_csv(performance_stat_path_ann, index=None)
