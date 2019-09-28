from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def setup_knn(features_train, labels_train, params=None):
    """
    takes datasets, fits, and predicts, using an KNeighborsClassifier
    :param features_train: training set
    :param labels_train: training labels
    :param features_test: test set
    :param labels_test: test labels
    :return: classifier, precision, and recall scores
    uses:

    """
    if params:
        clf = KNeighborsClassifier(**params)
    else:
        clf = KNeighborsClassifier()
    return clf


def fit_knn(clf, features_train, labels_train):
    # fit:
    clf.fit(features_train, labels_train)
    return


def predict_knn(clf, features_test):
    # predict:
    pred = clf.predict(features_test)
    return pred


def get_performance_knn(clf, pred, features_test, labels_test):
    knn_score = balanced_accuracy_score(labels_test, pred)

    knn_precision = precision_score(labels_test, pred, average='micro')

    knn_recall = recall_score(labels_test, pred, average='weighted')

    knn_f1 = f1_score(labels_test, pred, average='micro')

    return knn_score, knn_precision, knn_recall, knn_f1

