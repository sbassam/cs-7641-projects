from sklearn.decomposition import PCA, FastICA


def run_pca(X_train):
    pca = PCA(random_state=69)
    X_train_transformed = pca.fit_transform(X_train)
    return pca, X_train_transformed


def run_ica(X_train):
    ica = FastICA(random_state=69)
    X_train_transformed = ica.fit_transform(X_train)
    return ica, X_train_transformed
