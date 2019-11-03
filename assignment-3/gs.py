import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import time


def run_gs(X_train, y_train, ds_name, cluster_type=None, dr_type=None):
    gs_csv_path = 'out/gs-ann-results-' + ds_name + '-'
    if cluster_type:
        gs_csv_path += cluster_type
        gs_csv_path += '-'
    if dr_type:
        gs_csv_path += dr_type
    gs_csv_path += '.csv'
    alpha_arr = [.00001, .0001, .001, .01 , .1]
    hidden_layer_size_arr =[(i, i, i) for i in range(50, 300, 50)]
    # learning_rate_init = [0.0001, .001, .01, .1]
    # max_iter = [100, 200, 500, 1000]
    parameters = {'hidden_layer_sizes': hidden_layer_size_arr, 'alpha': alpha_arr}
                  #'learning_rate_init':learning_rate_init, 'max_iter':max_iter}
    clf = MLPClassifier(random_state=69)
    gs_clf = GridSearchCV(clf, parameters, cv=5, scoring='balanced_accuracy')
    gs_clf.fit(X_train, y_train)
    df = pd.DataFrame(gs_clf.cv_results_)
    df.to_csv(gs_csv_path, index=None)
    clf = MLPClassifier(random_state=69, **gs_clf.best_params_)
    return clf

