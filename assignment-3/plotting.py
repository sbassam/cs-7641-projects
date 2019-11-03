from datetime import datetime

import pandas as pd

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture


def plot_elbow(dsname, algname, X_train, min=2, max=13):
    if algname == 'KMeans':
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(min, max), title="Distortion Score Elbow for " + algname + " Clustering on " + dsname)
    if algname == 'GaussianMixture':
        return
        # model = mixture.GaussianMixture(covariance_type='full')
        # visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

    visualizer.fit(X_train)
    visualizer.show(outpath="images/elbow-"+dsname+"-"+algname+"-.png", clear_figure=True)

    return visualizer.elbow_value_

def plot_sil_score(X_train, ds_name, alg_name, kmax=13):
    # """https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb"""
    """https://www.scikit-yb.org/en/latest/api/cluster/silhouette.html"""
    sil = []
    for k in range(2, kmax+1):
        # kmeans = KMeans(n_clusters=k).fit(X_train)
        # labels = kmeans.labels_
        # sil.append(silhouette_score(X_train, labels, metric='euclidean'))


        model = KMeans(k, random_state=42)
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
        visualizer.fit(X_train)
        sil.append(visualizer.silhouette_score_)
        visualizer.show(outpath="images/sil-score-" + ds_name+ "-" + alg_name+"-"+str(k)+"-clusters.png", clear_figure=True)
    return sil.index(max(sil)) + 2



def plot_BIC(X_train, max_n_components=10):
    """source: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py"""


    n_samples = X_train.shape[0]

    # Generate random sample, two components
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    # X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
    #           .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
    X = X_train

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            qq = gmm.score(X)
            bic.append(qq)
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
           .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(X)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                               color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: full model, 2 components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()

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