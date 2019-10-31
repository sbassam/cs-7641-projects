from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import time
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import make_scorer, f1_score, accuracy_score
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
from gm import run_gm
from plotting import plot_elbow, plot_pca, plot_ica

ds_name = 'wine-quality'
#ds_name = 'abalone-ternary'
#ds_name = 'iris'

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

# plot elbow
#n_clusters = plot_elbow(ds_name, 'KMeans', X_train)

# run Gaussian Mixture
#run_gm(X_train, n_clusters)


# Dimensionality Reduction
# PCA
X_train_pca = X_train.copy()
pca, X_train_pca = run_pca(X_train_pca)
plot_pca(ds_name, pca)

# ICA
X_train_ica = X_train.copy()
ica, X_train_ica = run_ica(X_train_ica)
plot_ica(ds_name, ica)

n_clusters = plot_elbow(ds_name, 'KMeans', X_train_pca)
n_clusters = plot_elbow(ds_name, 'KMeans', X_train_ica)









