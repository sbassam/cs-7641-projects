from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier


def setup_dt(features_train, labels_train, params = None):
    """
    takes datasets, fits, and predicts, using an DecisionTreeClassifier
    :param params: dict, hyper parameters to use for the clf
    :param features_train: training set
    :param labels_train: training labels
    :return: classifier, precision, and recall scores
    uses:

    """

    if params:

        clf = DecisionTreeClassifier(**params)
    else:
        clf = DecisionTreeClassifier()  # max_depth=20, min_samples_leaf=40, max_leaf_nodes=20)
    return clf


def fit_dt(clf, features_train, labels_train):
    # fit:
    clf.fit(features_train, labels_train)
    return


def predict_dt(clf, features_test):
    # predict:
    pred = clf.predict(features_test)
    return pred


def get_performance_dt(clf, pred, features_test, labels_test):
    dt_score = balanced_accuracy_score(labels_test, pred)

    dt_precision = precision_score(labels_test, pred, average='micro')

    dt_recall = recall_score(labels_test, pred, average='weighted')

    dt_f1 = f1_score(labels_test, pred, average='micro')

    return dt_score, dt_precision, dt_recall, dt_f1