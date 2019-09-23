from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.svm import SVC


def setup_svm(features_train, labels_train, param=None):
    """
    takes datasets, fits, and predicts, using an MLPClassifier
    :param features_train: training set
    :param labels_train: training labels
    :param features_test: test set
    :param labels_test: test labels
    :return: classifier, precision, and recall scores
    uses:
    https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
    """


    if param:
        clf = SVC(**param)
    else:
        clf = SVC()

    return clf


def fit_svm(clf, features_train, labels_train):
    # fit:
    clf.fit(features_train, labels_train)
    return


def predict_svm(clf, features_test):
    # predict:
    pred = clf.predict(features_test)
    return pred


def get_performance_svm(clf, pred, features_test, labels_test):
    svm_score = balanced_accuracy_score(labels_test, pred)

    svm_precision = precision_score(labels_test, pred, average='micro')

    svm_recall = recall_score(labels_test, pred, average='weighted')

    svm_f1 = f1_score(labels_test, pred, average='micro')

    return svm_score, svm_precision, svm_recall, svm_f1
