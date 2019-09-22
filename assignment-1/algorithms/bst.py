from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def setup_bst(features_train, labels_train, param=None):
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
        clf = AdaBoostClassifier(**param)
    else:
        clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=7, min_samples_leaf = 2))

    return clf


def fit_bst(clf, features_train, labels_train):
    # fit:
    clf.fit(features_train, labels_train)
    return


def predict_bst(clf, features_test):
    # predict:
    pred = clf.predict(features_test)
    return pred


def get_performance_bst(clf, pred, features_test, labels_test):
    bst_score = balanced_accuracy_score(labels_test, pred)

    bst_precision = precision_score(labels_test, pred, average='micro')

    bst_recall = recall_score(labels_test, pred, average='weighted')

    bst_f1 = f1_score(labels_test, pred, average='micro')

    return bst_score, bst_precision, bst_recall, bst_f1
