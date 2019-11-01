from datetime import datetime

import pandas as pd
from sklearn import mixture
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools


def plot_elbow(dsname, algname, X_train, min=2, max=13):
    if algname == 'KMeans':
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(min, max), title="Distortion Score Elbow for " + algname + " Clustering on " + dsname)
    if algname == 'GaussianMixture':
        return
        # model = mixture.GaussianMixture(covariance_type='full')
        # visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

    visualizer.fit(X_train)
    visualizer.show()

    return visualizer.elbow_value_


def plot_pca(ds_name, pca):
    x = np.arange(1, pca.n_components_+1)
    y = pca.explained_variance_

    fig, ax = plt.subplots()
    ax.bar(x, y, color="b")
    for i, j in zip(x, y):
        ax.annotate(str("%.4f"%j), xy=(i-.2, j+.001), fontsize = 8)
    plt.xticks(x)
    plt.ylabel('Explained Variance', fontsize=14)
    plt.xlabel('Component', fontsize=14)
    plt.title('Explained Variance for PCA components - dataset: ' + ds_name,
              fontsize=14)
    plt.legend()
    plt.savefig(
        'images/' + ds_name + '-pca-explained-variance' + str(datetime.now()) + '.png')
    plt.close()
    return plt

def plot_ica(ds_name, ica):
    df = pd.DataFrame(ica._FastICA__sources)
    y = abs(df.kurtosis(axis = 0))
    x = np.arange(1, ica.components_.shape[0]+1)


    fig, ax = plt.subplots()
    ax.bar(x, y, color="b")
    for i, j in zip(x, y):
        ax.annotate(str("%.4f"%j), xy=(i-.2, j+.001), fontsize = 8)
    plt.xticks(x)
    plt.ylabel('Kurtosis', fontsize=14)
    plt.xlabel('Component', fontsize=14)
    plt.title('Kurtosis for ICA components - dataset: ' + ds_name,
              fontsize=14)
    plt.legend()
    plt.savefig(
        'images/' + ds_name + '-ica-kurtosis' + str(datetime.now()) + '.png')
    plt.close()
    return plt


def plot_ica_avg_kurtosis(ds_name, kurtosis_list):
    x = range(1, len(kurtosis_list)+1)
    y = kurtosis_list
    fig, ax = plt.subplots()
    ax.bar(x, y, color="b")
    for i, j in zip(x, y):
        ax.annotate(str("%.4f" % j), xy=(i - .2, j + .001), fontsize=8)
    plt.xticks(x)
    plt.ylabel('Average Kurtosis', fontsize=14)
    plt.xlabel('Component', fontsize=14)
    plt.title('Average Kurtosis vs # of components - dataset: ' + ds_name,
              fontsize=14)
    plt.legend()
    plt.savefig(
        'images/' + ds_name + '-ica-avgkurtosis-n-components' + str(datetime.now()) + '.png')
    plt.close()
    return plt