import time

import pandas as pd
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, adjusted_rand_score, mutual_info_score


def run_km(X_train, y_train, n_clusters):
    # analyze various metrics vs. number of clusters.
    csv_path = 'out/km-score-vs-k.csv'
    data = []
    cols = ['n_clusters', 'v_measure_score', 'adjusted_rand_score', 'AMI_score', 'fit_time']
    n_clusters_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 32, 64]
    for i in n_clusters_list:
        model = KMeans(n_clusters=i, random_state=69)
        start = time.time()
        model.fit(X_train)
        end = time.time()
        v_score = v_measure_score(y_train.ravel(), model.labels_)
        rand_score = adjusted_rand_score(y_train.ravel(), model.labels_)
        ami = mutual_info_score(y_train.ravel(), model.labels_)
        row = [i, v_score, rand_score, ami, end-start]
        data.append(row)
    result = pd.DataFrame(data, columns=cols)
    result.to_csv(csv_path, index=None)


    return

def run_gm(X_train, n_components):
    model = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
    model.fit_predict(X_train)
    return
