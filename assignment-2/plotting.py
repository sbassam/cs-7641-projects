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
    fig.savefig('images/' + problem_name + str(input_size) + '-fitness-vs-iters-' + str(datetime.now()) + '.png')
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
    fig.savefig('images/' + problem_name + str(input_size) + '-time-vs-iters-' + str(datetime.now()) + '.png')
    fig.clf()
    return plt


def plot_fitness_time(csv_path_list, method_names, problem_name, input_size):
    for i in range(len(csv_path_list)):
        df = pd.read_csv(csv_path_list[i])
        means = df[['Iteration', 'Fitness', 'Time']].groupby(['Iteration']).agg({'Fitness': np.mean, 'Time': np.mean})
        stds = df[['Iteration', 'Fitness', 'Time']].groupby(['Iteration']).agg({'Fitness': np.std, 'Time': np.std})

        plt.fill_between(means['Time'].values, means['Fitness'].values - .5 * stds['Fitness'].values,
                         means['Fitness'].values + .5 * stds['Fitness'].values, alpha=0.2)
        plt.plot(means['Time'].values, means['Fitness'].values, '--', color='C' + str(i),
                 label=method_names[i])
    plt.xscale('log', basex=10)
    plt.ylabel('Fitness score', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.title(problem_name + ' ' + str(input_size) + ' - Mean Fitnes vs. Mean Time')
    plt.legend()
    plt.savefig('images/' + problem_name + str(input_size) + '-fitness-vs-time-' + str(datetime.now()) + '.png')
    plt.close()


def plot_nn_score_vs_iters(alg_name):
    df = pd.read_csv('out/nn_' + alg_name + '.csv')


    plt.plot(df['iter'], df['train_accuracy'], 'x-', color="r",
             label="Training Accuracy")
    plt.plot(df['iter'], df['test_accuracy'], 'x-', color="g",
             label="Test Accuracy")
    plt.plot(df['iter'], df['test_f1'], '--', color="g",
             label="Test F1")

    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Iteration', fontsize=14)
    plt.title('Score vs. Iteration for training and test data' + alg_name,
              fontsize=14)
    plt.legend()
    plt.savefig(
        'images/' + alg_name + 'nn-score-vs-iteration-' + str(datetime.now()) + '.png')
    plt.close()
    return plt


def plot_nn_fit_time_vs_iter(alg_name):
    df = pd.read_csv('out/nn_' + alg_name + '.csv')


    plt.plot(df['iter'], df['fit_time'], 'o-', color="b")

    plt.ylabel('Fit Time', fontsize=14)
    plt.xlabel('Iteration', fontsize=14)
    plt.title(alg_name, fontsize=14)
    plt.savefig('images/' + alg_name + 'nn-fittime-vs-time-' + str(datetime.now()) + '.png')
    plt.close()
    return plt

def plot_nn_loss_vs_iter(alg_name):
    df = pd.read_csv('out/nn_' + alg_name + '.csv')

    plt.style.use('seaborn')

    plt.plot(df['iter'], df['loss'], 'o-', color="y")

    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Iteration', fontsize=14)
    plt.title(alg_name, fontsize=14)
    plt.savefig('images/' + alg_name + 'nn-loss-vs-iteration-' + str(datetime.now()) + '.png')
    plt.close()
    return plt