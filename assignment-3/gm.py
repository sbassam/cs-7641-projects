from sklearn import mixture


def run_gm(X_train, n_components):
    model = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
    model.fit_predict(X_train)
    return
