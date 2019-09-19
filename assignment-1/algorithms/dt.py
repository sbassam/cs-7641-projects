from sklearn.metrics import precision_score, recall_score, f1_score
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
    # f = open("performance_results.txt", "a")
    # score:
    dt_score = clf.score(features_test, labels_test)
    #
    # statement = "\nDecision Tree Score: " + str(dt_score)
    # f.write(statement)

    dt_precision = precision_score(labels_test, pred, average='micro')
    #
    # statement = "\nDecision Tree Precision: " + str(dt_precision)
    # f.write(statement)

    dt_recall = recall_score(labels_test, pred, average='weighted')
    #
    # statement = "\nDecision Tree Recall: " + str(dt_recall) + "\n"
    # f.write(statement)

    dt_f1 = f1_score(labels_test, pred, average='weighted')

    # statement = "\nDecision Tree F1: " + str(dt_f1) + "\n-------------------\n"
    # f.write(statement)
    # f.close()

    return dt_score, dt_precision, dt_recall, dt_f1
