from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier


def setup_ann(features_train, labels_train, features_test, labels_test):
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

    clf = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                        beta_2=0.999, early_stopping=False, epsilon=1e-08,
                        hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
                        learning_rate_init=0.001, max_iter=500, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=None,
                        shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
                        verbose=False, warm_start=False)
    return clf


def fit_ann(clf, features_train, labels_train):
    # fit:
    clf.fit(features_train, labels_train)
    return


def predict_ann(clf, features_test):
    # predict:
    pred = clf.predict(features_test)
    return pred


def get_performance_ann(clf, pred, features_test, labels_test):
    # score:
    nn_score = clf.score(features_test, labels_test)
    print(nn_score)
    nn_precision = precision_score(labels_test, pred, average='micro')
    print('precision for nn: ', nn_precision)
    nn_recall = recall_score(labels_test, pred, average='weighted')
    print('recall for nn: ', nn_recall)
    nn_f1 = f1_score(labels_test, pred, average='weighted')
    print('f1 for nn: ', nn_f1)

    return nn_precision, nn_recall, nn_f1