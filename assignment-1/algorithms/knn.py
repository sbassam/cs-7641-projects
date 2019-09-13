from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier



def setup_knn(features_train, labels_train, features_test, labels_test):
    """
    takes datasets, fits, and predicts, using an KNeighborsClassifier
    :param features_train: training set
    :param labels_train: training labels
    :param features_test: test set
    :param labels_test: test labels
    :return: classifier, precision, and recall scores
    uses:

    """

    clf = KNeighborsClassifier(n_neighbors=15)
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
    # score:
    knn_score = clf.score(features_test, labels_test)
    print(knn_score)
    knn_precision = precision_score(labels_test, pred, average='micro')
    print('precision for knn: ', knn_precision)
    knn_recall = recall_score(labels_test, pred, average='weighted')
    print('recall for knn: ', knn_recall)
    knn_f1 = f1_score(labels_test, pred, average='weighted')
    print('f1 for knn: ', knn_f1)

    return knn_precision, knn_recall, knn_f1
