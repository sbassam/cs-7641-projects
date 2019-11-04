import pandas as pd
import scprep as scprep
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from numpy import linalg as LA
import numpy as np


def run_pca(X_train, X_test=None):
    pca = PCA(random_state=69)
    X_train_transformed = pca.fit_transform(X_train)
    if X_test is not None:
        X_test_transformed = pca.transform(X_test)
        return pca, X_train_transformed, X_test_transformed
    return pca, X_train_transformed


def run_ica(X_train, X_test=None):
    kurtosis_means = []
    kurtosis_stds = []
    for i in range(1, X_train.shape[1] + 1):
        ica = FastICA(random_state=69, n_components=i)
        X_train_transformed = ica.fit_transform(X_train)
        df = pd.DataFrame(X_train_transformed)
        kurtosis_mean = abs(df.kurtosis(axis=0)).mean()
        kurtosis_std = abs(df.kurtosis(axis=0)).std() * .1
        kurtosis_means.append(kurtosis_mean)
        kurtosis_stds.append(kurtosis_std)
    max_kurtosis = max(kurtosis_means)
    max_index = kurtosis_means.index(max_kurtosis)

    ica = FastICA(random_state=69, n_components=max_index + 1)
    X_train_transformed = ica.fit_transform(X_train)

    if X_test is not None:
        X_test_transformed = ica.transform(X_test)
        return ica, X_train_transformed, X_test_transformed, kurtosis_means, kurtosis_stds

    return ica, X_train_transformed, kurtosis_means, kurtosis_std


def run_svd(X_train, X_test=None):
    loss_means = []
    loss_stds = []
    for i in range(2, X_train.shape[1]):
        ith_component = []  # list of losses for the n_components=i
        for j in range(20):


            svd = TruncatedSVD(random_state=69 + j, n_components=i)
            svd_result = svd.fit_transform(X_train)

            svd_proj_back = svd.inverse_transform(svd_result)
            total_loss = LA.norm((X_train - svd_proj_back), None)
            ith_component.append(total_loss)
        ith_component = np.array(ith_component)
        ith_mean = np.mean(ith_component, axis=0)
        ith_std = np.std(ith_component, axis=0)
        loss_means.append(ith_mean)
        loss_stds.append(ith_std)
    min_loss = min(loss_means)
    min_ind = loss_means.index(min_loss)

    svd = SparseRandomProjection(random_state=69, n_components=min_ind+1)
    X_train_transformed = svd.fit_transform(X_train)

    if X_test is not None:
        X_test_transformed = svd.transform(X_test)
        return svd, X_train_transformed, X_test_transformed, loss_means, loss_stds
    return svd, X_train_transformed, loss_means, loss_stds

def run_rp(X_train, X_test=None):
    loss_means = []
    loss_stds = []
    for i in range(2, X_train.shape[1]):
        ith_component = []  # list of losses for the n_components=i
        for j in range(20):

            #rp = GaussianRandomProjection(random_state=69 + j, n_components=i)
            rp = scprep.reduce.InvertibleRandomProjection(random_state=69 + j, n_components=i)
            rp_result = rp.fit_transform(X_train)
            #IRP = scprep.reduce.SparseInputPCA(rp)
            rp_proj_back = rp.inverse_transform(rp_result)
            total_loss = LA.norm((X_train - rp_proj_back), None)
            ith_component.append(total_loss)
        ith_component = np.array(ith_component)
        ith_mean = np.mean(ith_component, axis=0)
        ith_std = np.std(ith_component, axis=0)
        loss_means.append(ith_mean)
        loss_stds.append(ith_std)
    min_loss = min(loss_means)
    min_ind = loss_means.index(min_loss)

    rp = scprep.reduce.InvertibleRandomProjection(random_state=69, n_components=min_ind+1)
    X_train_transformed = rp.fit_transform(X_train)

    if X_test is not None:
        X_test_transformed = rp.transform(X_test)
        return rp, X_train_transformed, X_test_transformed, loss_means, loss_stds
    return rp, X_train_transformed, loss_means, loss_stds
