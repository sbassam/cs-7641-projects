from sklearn.metrics import precision_score, recall_score, f1_score
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
    # f = open("performance_results.txt", "a")
    # score:
    knn_score = clf.score(features_test, labels_test)

    # statement = "\nDecision Tree Score: " + str(knn_score)
    # f.write(statement)

    knn_precision = precision_score(labels_test, pred, average='micro')
    #
    # statement = "\nDecision Tree Precision: " + str(knn_precision)
    # f.write(statement)

    knn_recall = recall_score(labels_test, pred, average='weighted')
    #
    # statement = "\nDecision Tree Recall: " + str(knn_recall) + "\n"
    # f.write(statement)

    knn_f1 = f1_score(labels_test, pred, average='weighted')

    # statement = "\nDecision Tree F1: " + str(knn_f1) + "\n-------------------\n"
    # f.write(statement)
    # f.close()

    return knn_score, knn_precision, knn_recall, knn_f1

