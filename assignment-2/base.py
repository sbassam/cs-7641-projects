# 1) plot the result of hyperparameter tuning:
import sys

import flipflop
import knapsack
import tsp
import fourpeaks
from plotting import plot_hp_sa, plot_hp_rhc, plot_hp_ga, plot_hp_mimic, plot_fitness_iter, plot_time_iter, \
    plot_fitness_time


# Hyper parameter Tuning

# plot_hp_rhc('Flipflop', 'rhc_flipflop.csv')
# plot_hp_sa('Flipflop', 'sa_flipflop.csv')
# plot_hp_ga('Flipflop', 'ga_flipflop.csv')
# plot_hp_mimic('Flipflop', 'mimic_flipflop.csv')

# plot_hp_rhc('FourPeaks', 'rhc_FourPeaks.csv')
# plot_hp_sa('FourPeaks', 'sa_FourPeaks.csv')
# plot_hp_ga('FourPeaks', 'ga_FourPeaks.csv')
# plot_hp_mimic('FourPeaks', 'mimic_FourPeaks.csv')
#
# plot_hp_rhc('TSP', 'rhc_tsp.csv', convert_to_max=True)
# plot_hp_sa('TSP', 'sa_tsp.csv', convert_to_max=True)
# plot_hp_ga('TSP', 'ga_tsp.csv', convert_to_max=True)
# #plot_hp_mimic('TSP', 'mimic_tsp.csv', convert_to_max=True)
#
# plot_hp_rhc('Knapsack', 'rhc_Knapsack.csv')
# plot_hp_sa('Knapsack', 'sa_Knapsack.csv')
# plot_hp_ga('Knapsack', 'ga_Knapsack.csv')
# plot_hp_mimic('Knapsack', 'mimic_Knapsack.csv')

def optimize_knapsack(input_size=50):
    restart = 1
    temperature = 10
    ga_pop_size = 2500
    mut_rate = 0.2
    mimic_pop_size = 3000
    keep_pct = 0.2
    knapsack_csv_list = []

    # use default value for other params
    knapsack_csv_list.append(knapsack.run_rhc_knapsack(input_size=input_size, restart=restart))
    knapsack_csv_list.append(knapsack.run_sa_knapsack(input_size=input_size, temperature=temperature))
    knapsack_csv_list.append(knapsack.run_ga_knapsack(input_size=input_size, mut_rate=mut_rate, pop_size=ga_pop_size))
    knapsack_csv_list.append(
        knapsack.run_mimic_knapsack(input_size=input_size, keep_pct=keep_pct, pop_size=mimic_pop_size))
    # plot
    plot_fitness_iter(csv_path_list=knapsack_csv_list, method_names=['rhc', 'sa', 'ga', 'mimic'],
                      problem_name='Knapsack', input_size=input_size)
    plot_time_iter(csv_path_list=knapsack_csv_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='Knapsack',
                      input_size=input_size)
    plot_fitness_time(csv_path_list=knapsack_csv_list, method_names=['rhc', 'sa', 'ga', 'mimic'],
                      problem_name='Knapsack', input_size=input_size)


def optimize_tsp(input_size=10):
    restart = 1
    temperature = 10
    ga_pop_size = 2500
    mut_rate = 0.2
    mimic_pop_size = 3000
    keep_pct = 0.2
    tsp_csv_list = []

    # use default value for other params
    tsp_csv_list.append(tsp.run_rhc_tsp(input_size=input_size, restart=restart))
    tsp_csv_list.append(tsp.run_sa_tsp(input_size=input_size, temperature=temperature))
    tsp_csv_list.append(tsp.run_ga_tsp(input_size=input_size, mut_rate=mut_rate, pop_size=ga_pop_size))
    tsp_csv_list.append(tsp.run_mimic_tsp(input_size=input_size, keep_pct=keep_pct, pop_size=mimic_pop_size))
    # plot
    plot_fitness_iter(csv_path_list=tsp_csv_list, method_names=['rhc', 'sa', 'ga', 'mimic'],
                      problem_name='TSP', input_size=input_size)
    plot_time_iter(csv_path_list=tsp_csv_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='TSP',
                   input_size=input_size)
    plot_fitness_time(csv_path_list=tsp_csv_list, method_names=['rhc', 'sa', 'ga', 'mimic'],
                      problem_name='TSP', input_size=input_size)


def optimize_fourpeaks(input_size=50):
    restart = 1
    temperature = 10
    ga_pop_size = 2500
    mut_rate = 0.2
    mimic_pop_size = 3000
    keep_pct = 0.2
    fourpeaks_csv_list = []

    # use default value for other params
    fourpeaks_csv_list.append(fourpeaks.run_rhc_fourpeaks(input_size=input_size, restart=restart))
    fourpeaks_csv_list.append(fourpeaks.run_sa_fourpeaks(input_size=input_size, temperature=temperature))
    fourpeaks_csv_list.append(
        fourpeaks.run_ga_fourpeaks(input_size=input_size, mut_rate=mut_rate, pop_size=ga_pop_size))
    fourpeaks_csv_list.append(
        fourpeaks.run_mimic_fourpeaks(input_size=input_size, keep_pct=keep_pct, pop_size=mimic_pop_size))
    # plot
    plot_fitness_iter(csv_path_list=fourpeaks_csv_list, method_names=['rhc', 'sa', 'ga', 'mimic'],
                      problem_name='Four Peaks', input_size=input_size)
    plot_time_iter(csv_path_list=fourpeaks_csv_list, method_names=['rhc', 'sa', 'ga', 'mimic'],
                   problem_name='Four Peaks',
                   input_size=input_size)
    plot_fitness_time(csv_path_list=fourpeaks_csv_list, method_names=['rhc', 'sa', 'ga', 'mimic'],
                      problem_name='Four Peaks', input_size=input_size)



def optimize_flipflop(input_size=50):
    restart = 1
    temperature = 10
    ga_pop_size = 2500
    mut_rate = 0.2
    mimic_pop_size = 3000
    keep_pct = 0.2
    flipflop_csv_list = []

    # use default value for other params
    flipflop_csv_list.append(flipflop.run_rhc_flipflop(input_size=input_size, restart=restart))
    flipflop_csv_list.append(flipflop.run_sa_flipflop(input_size=input_size, temperature=temperature))
    flipflop_csv_list.append(flipflop.run_ga_flipflop(input_size=input_size, mut_rate=mut_rate, pop_size=ga_pop_size))
    flipflop_csv_list.append(
        flipflop.run_mimic_flipflop(input_size=input_size, keep_pct=keep_pct, pop_size=mimic_pop_size))
    # plot
    plot_fitness_iter(csv_path_list=flipflop_csv_list, method_names=['rhc', 'sa', 'ga', 'mimic'],
                      problem_name='Flip Flop', input_size=input_size)
    plot_time_iter(csv_path_list=flipflop_csv_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='Flip Flop',
                   input_size=input_size)
    plot_fitness_time(csv_path_list=flipflop_csv_list, method_names=['rhc', 'sa', 'ga', 'mimic'],
                      problem_name='Flip Flop', input_size=input_size)


if __name__ == '__main__':
    input_size = int(sys.argv[2])
    if sys.argv[1] == 'knapsack':
        optimize_knapsack(input_size=input_size)
    if sys.argv[1] == 'tsp':
        optimize_tsp(input_size=input_size)
    if sys.argv[1] == 'fourpeaks':
        optimize_fourpeaks(input_size=input_size)
    if sys.argv[1] == 'flipflop':
        optimize_flipflop(input_size=input_size)
