import pandas as pd
from sklearn.decomposition import PCA, FastICA


def run_pca(X_train):
    pca = PCA(random_state=69)
    X_train_transformed = pca.fit_transform(X_train)
    return pca, X_train_transformed


def run_ica(X_train, X_test=None):
    kurtosis_list =  []
    for i in range(1, X_train.shape[1]+1):
        ica = FastICA(random_state=69, n_components=i)
        X_train_transformed = ica.fit_transform(X_train)
        df = pd.DataFrame(X_train_transformed)
        kurtosis = abs(df.kurtosis(axis=0)).mean()
        kurtosis_list.append(kurtosis)
    max_kurtosis = max(kurtosis_list)
    max_index = kurtosis_list.index(max_kurtosis)

    ica = FastICA(random_state=69, n_components=max_index+1)
    X_train_transformed = ica.fit_transform(X_train)

    if X_test is not None:
        X_test_transformed = ica.transform(X_test)
        return ica, X_train_transformed, X_test_transformed, kurtosis_list


    return ica, X_train_transformed, kurtosis_list

