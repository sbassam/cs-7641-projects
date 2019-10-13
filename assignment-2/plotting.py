import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time

from sklearn.model_selection import GridSearchCV


def plot_hp_rhc(opt_name, filename, no_of_runs=5, convert_to_max=False):
    rhc_tuning = pd.read_csv('out/' + filename)
    if convert_to_max:
        # Adjust the fitness to convert to a maximization problem
        rhc_tuning['Fitness'] = 1000 * rhc_tuning['Fitness'] ** -1
    # group by restart and Iteration
    rhc_grouped = rhc_tuning[['Fitness', 'Restarts', 'Iteration', 'run_number']]
    rhc_grouped = rhc_grouped.groupby(['Restarts', 'Iteration'])
    # Take the mean of runs
    rhc_grouped = rhc_grouped.agg({'Fitness': np.mean})
    # https://scentellegher.github.io/programming/2017/07/15/pandas-groupby-multiple-columns-plot.html
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.xscale('log', basex=2)
    plt.ylabel('Mean Fitness over ' + str(no_of_runs) + ' runs', fontsize=14)
    plt.title(opt_name + ': RHC Tuning', fontsize=14)
    # use unstack()
    rhc_grouped.groupby(['Iteration', 'Restarts']).mean()['Fitness'].unstack().plot(ax=ax)

    plt.savefig('images/' + opt_name + '-hp-rhc-' + str(datetime.now()) + '.png')
    plt.close()


def plot_hp_sa(opt_name, filename, no_of_runs=5, convert_to_max=False):
    sa_tuning = pd.read_csv('out/' + filename)
    if convert_to_max:
        # Adjust the fitness to convert to a maximization problem
        sa_tuning['Fitness'] = 1000 * sa_tuning['Fitness'] ** -1
    # group by Temperature and Iteration
    sa_grouped = sa_tuning[['Fitness', 'Temperature', 'Iteration', 'run_number']]
    sa_grouped = sa_grouped.groupby(['Temperature', 'Iteration'])
    # Take the mean of runs
    sa_grouped = sa_grouped.agg({'Fitness': np.mean})
    # https://scentellegher.github.io/programming/2017/07/15/pandas-groupby-multiple-columns-plot.html
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.xscale('log', basex=2)
    plt.ylabel('Mean Fitness over ' + str(no_of_runs) + ' runs', fontsize=14)
    plt.title(opt_name + ': SA Tuning', fontsize=14)
    # use unstack()
    sa_grouped.groupby(['Iteration', 'Temperature']).mean()['Fitness'].unstack().plot(ax=ax)
    plt.savefig('images/' + opt_name + '-hp-sa-' + str(datetime.now()) + '.png')
    plt.close()


def plot_hp_ga(opt_name, filename, no_of_runs=5, convert_to_max=False):
    ga_tuning = pd.read_csv('out/' + filename)
    if convert_to_max:
        # Adjust the fitness to convert to a maximization problem
        ga_tuning['Fitness'] = 1000 * ga_tuning['Fitness'] ** -1
    # https://www.geeksforgeeks.org/join-two-text-columns-into-a-single-column-in-pandas/
    ga_tuning['Population Size - Mutation Rate'] = ga_tuning['Population Size'].astype(str) + "-" \
                                                   + ga_tuning['Mutation Rate'].astype(str)
    # group by Temperature and Iteration
    ga_grouped = ga_tuning[['Fitness', 'Population Size - Mutation Rate', 'Iteration', 'run_number']]
    ga_grouped = ga_grouped.groupby(['Population Size - Mutation Rate', 'Iteration'])
    # Take the mean of runs
    ga_grouped = ga_grouped.agg({'Fitness': np.mean})
    # https://scentellegher.github.io/programming/2017/07/15/pandas-groupby-multiple-columns-plot.html
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.xscale('log', basex=2)
    plt.ylabel('Mean Fitness over ' + str(no_of_runs) + ' runs', fontsize=14)
    plt.title(opt_name + ': GA Tuning', fontsize=14)
    # use unstack()
    ga_grouped.groupby(['Iteration', 'Population Size - Mutation Rate']).mean()['Fitness'].unstack().plot(ax=ax)
    plt.savefig('images/' + opt_name + '-hp-ga-' + str(datetime.now()) + '.png')
    plt.close()


def plot_hp_mimic(opt_name, filename, no_of_runs=5, convert_to_max=False):
    mimic_tuning = pd.read_csv('out/' + filename)
    if convert_to_max:
        # Adjust the fitness to convert to a maximization problem
        mimic_tuning['Fitness'] = 1000 * mimic_tuning['Fitness'] ** -1
    # https://www.geeksforgeeks.org/join-two-text-columns-into-a-single-column-in-pandas/
    mimic_tuning['Population Size - Keep %'] = mimic_tuning['pop_size'].astype(str) + "-" \
                                               + mimic_tuning['keep_pct'].astype(str)
    # group by Temperature and Iteration
    mimic_grouped = mimic_tuning[['Fitness', 'Population Size - Keep %', 'Iteration', 'run_number']]
    mimic_grouped = mimic_grouped.groupby(['Population Size - Keep %', 'Iteration'])
    # Take the mean of runs
    mimic_grouped = mimic_grouped.agg({'Fitness': np.mean})
    # https://scentellegher.github.io/programming/2017/07/15/pandas-groupby-multiple-columns-plot.html
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.xscale('log', basex=2)
    plt.ylabel('Mean Fitness over ' + str(no_of_runs) + ' runs', fontsize=14)
    plt.title(opt_name + ': mimic Tuning', fontsize=14)
    # use unstack()
    mimic_grouped.groupby(['Iteration', 'Population Size - Keep %']).mean()['Fitness'].unstack().plot(ax=ax)
    plt.savefig('images/' + opt_name + '-hp-mimic-' + str(datetime.now()) + '.png')
    plt.close()


def plot_fitness_iter(csv_path_list, method_names, problem_name, input_size):
    df = pd.DataFrame(columns=['Iteration', 'Fitness', 'opt_method'])
    for i in range(len(csv_path_list)):
        data = pd.read_csv(csv_path_list[i])
        num_runs = data[data['Iteration'] == 0].shape[0]
        data = data[['Iteration', 'Fitness']]
        data['opt_method'] = method_names[i] + '- ' + str(num_runs) + ' runs'
        df = df.append(data)
    ax = sns.lineplot(x="Iteration", y="Fitness", hue='opt_method', style='opt_method', data=df)
    ax.set_xscale('log', basex=2)
    ax.set_title(problem_name + ' ' + str(input_size) + ' - Mean fitness vs. iterations')
    fig = ax.get_figure()
    fig.savefig('images/' + problem_name + str(input_size)+ '-fitness-vs-iters-' + str(datetime.now()) + '.png')
    fig.clf()
    return plt

def plot_time_iter(csv_path_list, method_names, problem_name, input_size):
    df = pd.DataFrame(columns=['Iteration', 'Time', 'opt_method'])
    for i in range(len(csv_path_list)):
        data = pd.read_csv(csv_path_list[i])
        num_runs = data[data['Iteration'] == 0].shape[0]
        data = data[['Iteration', 'Time']]
        data['opt_method'] = method_names[i] + '- ' + str(num_runs) + ' runs'
        df = df.append(data)
    ax = sns.lineplot(x="Iteration", y="Time", hue='opt_method', style='opt_method', data=df)
    ax.set_xscale('log', basex=2)
    ax.set_title(problem_name + ' ' + str(input_size) + ' - Mean time vs. iterations')
    fig = ax.get_figure()
    fig.savefig('images/' + problem_name + str(input_size)+ '-time-vs-iters-' + str(datetime.now()) + '.png')
    fig.clf()
    return plt

def plot_fitness_time(csv_path_list, method_names, problem_name, input_size):
    for i in range(len(csv_path_list)):
        df = pd.read_csv(csv_path_list[i])
        means = df[['Iteration', 'Fitness', 'Time']].groupby(['Iteration']).agg({'Fitness': np.mean, 'Time': np.mean})
        stds = df[['Iteration', 'Fitness', 'Time']].groupby(['Iteration']).agg({'Fitness': np.std, 'Time': np.std})

        plt.fill_between(means['Time'].values, means['Fitness'].values - .5*stds['Fitness'].values,
                             means['Fitness'].values + .5*stds['Fitness'].values, alpha=0.2)
        plt.plot(means['Time'].values, means['Fitness'].values, '--', color='C'+str(i),
                     label=method_names[i])
    plt.xscale('log', basex=10)
    plt.ylabel('Fitness score', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.title(problem_name + ' ' + str(input_size) + ' - Mean Fitnes vs. Mean Time')
    plt.legend()
    plt.savefig('images/' + problem_name + str(input_size) + '-fitness-vs-time-' + str(datetime.now()) + '.png')
    plt.close()





def plot_learning_curve(clf, features_train, labels_train, clf_name, dataset_name,
                        train_sizes=np.linspace(0.01, 1.0, 5), cv=5, shuffle=True, scoring='balanced_accuracy'):
    """

    :param scoring:
    :param clf: estimator
    :param features_train: training set
    :param labels_train: labels for training set
    :param clf_name: string
    :param dataset_name: string the name of the dataset
    :param train_sizes: array of set sizes.
    :param cv: cross validation folds
    :param shuffle: whether or not to shuffle for the cross validation operation
    :return: the plt object

    uses:
    https://www.dataquest.io/blog/learning-curves-machine-learning/
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    """

    train_sizes_abs, train_scores, validation_scores = learning_curve(clf, features_train, labels_train,
                                                                      train_sizes=train_sizes, cv=cv,
                                                                      shuffle=shuffle, scoring=scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    plt.style.use('seaborn')

    plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes_abs, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes_abs, validation_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # plt.plot(train_sizes_abs, train_scores.mean(axis = 1), label = 'Training error')
    # plt.plot(train_sizes_abs, validation_scores.mean(axis = 1), label = 'Validation error')
    plt.ylabel('Balanced Accuracy Score', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves for ' + clf_name + ' classifier on ' + dataset_name, fontsize=14)
    plt.legend()
    plt.savefig('images/' + dataset_name + '-learning-curve-' + clf_name + '-' + str(datetime.now()) + '.png')
    plt.close()
    return plt





def plot_nn_time_vs_iters(features_train, labels_train, clf, clf_name, dataset_name, alg_name):
    x_arr = [1, 50, 100, 200, 400, 800, 1500, 3000, 6000, 12000]
    iter_data = pd.DataFrame()
    csv_path = 'out/nn_' + str(alg_name) + '.csv'
    for size in x_arr:
        start = time.time()
        parameters = {'max_iter': np.array([size])}  # {'max_iter': np.arange(200, 800, 100) }
        gc_clf = GridSearchCV(clf, parameters, cv=5, scoring='balanced_accuracy', return_train_score=True)
        gc_clf.fit(features_train, labels_train)
        finish = time.finish()
        print('max iter: ', size, '\n', 'algorithm: ', alg_name, '\n', 'time: ', finish-start)
    # for size in np.arange(.01, 1., .1):
    #     iter_train, iter_test, iter_y_train, iter_y_test = train_test_split(features_train, labels_train,
    #                                                                         train_size=size)
    #
    #     # gc_params = {'max_iter': np.array([5000])}  # {'max_iter': np.arange(200, 800, 100) }
    #     gc_clf = GridSearchCV(clf, gc_params, cv=5, scoring='balanced_accuracy')
    #     gc_clf.fit(iter_train, iter_y_train)
    #     x_arr.append(iter_train.shape[0])
    #     #     fit_time_list[iter_train.shape[0]]=gc_clf.cv_results_['mean_fit_time'][0]
    #     #     accuracy_list[iter_train.shape[0]]=gc_clf.cv_results_['mean_test_score'][0]
        iter_data = pd.concat([iter_data, pd.DataFrame(gc_clf.cv_results_)])
    iter_data.to_csv(csv_path, index=None)

    plt.style.use('seaborn')

    plt.fill_between(x_arr, iter_data.mean_fit_time - iter_data.std_fit_time,
                     iter_data.mean_fit_time + iter_data.std_fit_time, alpha=0.1,
                     color="r")

    plt.plot(x_arr, iter_data.mean_fit_time, 'o-', color="r",
             label="Iteration time")

    plt.ylabel('Mean Fit time (sec)', fontsize=14)
    plt.xlabel('Iteration', fontsize=14)
    plt.title(dataset_name + '- ' + 'Validation Curve for ' + clf_name + ' classifier, parameter: ',
              fontsize=14)
    plt.legend()
    plt.savefig(
        'images/' + dataset_name + 'time-vs-iteration-' + clf_name + '-'  + str(datetime.now()) + '.png')
    plt.close()
    return plt

def plot_accuracy_vs_iterations(features_train, labels_train, clf, clf_name, dataset_name, gc_params):
    x_arr = []
    iter_data = pd.DataFrame()
    for size in np.arange(.01, 1., .05):
        iter_train, iter_test, iter_y_train, iter_y_test = train_test_split(features_train, labels_train,
                                                                            train_size=size)

        # gc_params = {'max_iter': np.array([5000])}  # {'max_iter': np.arange(200, 800, 100) }
        gc_clf = GridSearchCV(clf, gc_params, cv=5, scoring='balanced_accuracy')
        gc_clf.fit(iter_train, iter_y_train)
        x_arr.append(iter_train.shape[0])
        #     fit_time_list[iter_train.shape[0]]=gc_clf.cv_results_['mean_fit_time'][0]
        #     accuracy_list[iter_train.shape[0]]=gc_clf.cv_results_['mean_test_score'][0]
        iter_data = pd.concat([iter_data, pd.DataFrame(gc_clf.cv_results_)])

    plt.style.use('seaborn')

    plt.fill_between(x_arr, iter_data.mean_test_score - iter_data.std_test_score,
                     iter_data.mean_test_score + iter_data.std_test_score, alpha=0.1,
                     color="r")

    plt.plot(x_arr, iter_data.mean_test_score, 'o-', color="r")

    plt.ylabel('Balanced Accuracy Score', fontsize=14)
    plt.xlabel('Training Size', fontsize=14)
    plt.title(dataset_name + '- ' + 'Validation Curve for ' + clf_name + ' classifier, parameter: ',
              fontsize=14)
    plt.legend()
    plt.savefig(
        'images/' + dataset_name + 'accuracy-vs-iteration-' + clf_name + '-'  + str(datetime.now()) + '.png')
    plt.close()
    return plt




