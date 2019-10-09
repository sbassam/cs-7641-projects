import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


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






