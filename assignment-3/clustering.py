import time
import numpy as np
import pandas as pd
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, adjusted_rand_score, mutual_info_score
from sklearn.preprocessing import OneHotEncoder


def run_km(ds_name, X_train, y_train, n_clusters, X_test=None, alg_name='KMeans'):
    # analyze various metrics vs. number of clusters.
    csv_path = 'out/'+ds_name+'-'+alg_name+'-score-vs-k.csv'
    data = []
    cols = ['ds_name', 'alg_name', 'n_clusters', 'v_measure_score', 'adjusted_rand_score', 'AMI_score', 'fit_time']
    n_clusters_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 32, 64]
    for i in n_clusters_list:
        model = KMeans(n_clusters=i, random_state=69)
        start = time.time()
        model.fit(X_train)
        end = time.time()
        v_score = v_measure_score(y_train.ravel(), model.labels_)
        rand_score = adjusted_rand_score(y_train.ravel(), model.labels_)
        ami = mutual_info_score(y_train.ravel(), model.labels_)
        row = [ds_name, alg_name, i, v_score, rand_score, ami, end-start]
        data.append(row)
    result = pd.DataFrame(data, columns=cols)
    result.to_csv(csv_path, index=None)

    # run for the given number of clusters
    model = KMeans(n_clusters=n_clusters, random_state=69)
    start = time.time()
    model.fit(X_train)
    end = time.time()
    v_score = v_measure_score(y_train.ravel(), model.labels_)
    rand_score = adjusted_rand_score(y_train.ravel(), model.labels_)
    ami = mutual_info_score(y_train.ravel(), model.labels_)
    result = [ds_name, alg_name, n_clusters, v_score, rand_score, ami, end - start]
    X_train_new = np.array(model.labels_).reshape(-1,1)
    one_hot = OneHotEncoder()
    X_train_new = one_hot.fit_transform(X_train_new).todense()

    if X_test is not None:
        pred = model.predict(X_test)
        X_test_new = np.array(pred).reshape(-1, 1)
        one_hot = OneHotEncoder()
        X_test_new = one_hot.fit_transform(X_test_new).todense()
        return result, X_train_new, X_test_new

    return result, X_train_new

def run_gm(ds_name, X_train, y_train, n_components, cv_type='full', X_test=None, alg_name='GM'):
    # analyze various metrics vs. number of clusters.
    csv_path = 'out/' + ds_name + '-' +alg_name + '-score-vs-n-components.csv'
    data = []
    cols = ['ds_name', 'alg_name', 'n_clusters', 'v_measure_score', 'adjusted_rand_score', 'AMI_score', 'fit_time']
    n_components_list = range(2, 15)
    for i in n_components_list:
        model = mixture.GaussianMixture(n_components=i, covariance_type=cv_type, random_state=69)
        start = time.time()
        pred = model.fit_predict(X_train)
        end = time.time()
        v_score = v_measure_score(y_train.ravel(), pred)
        rand_score = adjusted_rand_score(y_train.ravel(), pred)
        ami = mutual_info_score(y_train.ravel(), pred)
        row = [ds_name, alg_name, i, v_score, rand_score, ami, end - start]
        data.append(row)
    result = pd.DataFrame(data, columns=cols)
    result.to_csv(csv_path, index=None)

    # run for the given number of components
    model = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=69)
    start = time.time()
    pred = model.fit_predict(X_train)
    end = time.time()
    v_score = v_measure_score(y_train.ravel(), pred)
    rand_score = adjusted_rand_score(y_train.ravel(), pred)
    ami = mutual_info_score(y_train.ravel(), pred)
    result = [ds_name, alg_name, n_components, v_score, rand_score, ami, end - start]
    X_train_new = np.array(pred).reshape(-1, 1)
    one_hot = OneHotEncoder()
    X_train_new = one_hot.fit_transform(X_train_new).todense()

    if X_test is not None:
        pred = model.predict(X_test)
        X_test_new = np.array(pred).reshape(-1, 1)
        one_hot = OneHotEncoder()
        X_test_new = one_hot.fit_transform(X_test_new).todense()
        return result, X_train_new, X_test_new

    return result, X_train_new
