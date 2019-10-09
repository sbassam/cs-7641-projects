# 1) plot the result of hyperparameter tuning:
import sys

import knapsack
from plotting import plot_hp_sa, plot_hp_rhc, plot_hp_ga, plot_hp_mimic


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

def optimize_knapsack():
    restart = 1
    temperature = 10
    ga_pop_size = 2500
    mut_rate = 0.2
    mimic_pop_size = 3000
    keep_pct = 0.2

    # use default value for other params
    knapsack.run_rhc_knapsack(restart=restart)
    knapsack.run_sa_knapsack(temperature=10)
    knapsack.run_ga_knapsack(mut_rate=mut_rate, pop_size=ga_pop_size)
    knapsack.run_mimic_knapsack(keep_pct=keep_pct, pop_size=mimic_pop_size)


if __name__== '__main__':
    print ('test')
    if sys.argv[1] == 'knapsack':
        optimize_knapsack()
    else:
        print('wrong!')
