import sklearn
import pandas as pd
import numpy as np
from mlrose import FourPeaks, RHCRunner, SARunner, GARunner, MIMICRunner, DiscreteOpt

fitness = FourPeaks()
problem = DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)


##### Tune Hyperparameters #####
# 1. random hill climbing
rhc = RHCRunner(problem,
                experiment_name='rhc_run_1_len_100',
                output_directory='out/',
                seed=123,
                iteration_list=2 ** np.arange(14),
                restart_list=[1, 2, 4, 6, 8, 10],
                max_attempts=5000,
                generate_curves=True)
rhc_run_stats, rhc_run_curves = rhc.run()

# random hill climbing
rhc = RHCRunner(problem,
                experiment_name='rhc_run_1_len_100',
                output_directory='out/',
                seed=123,
                iteration_list=2 ** np.arange(14),
                restart_list=[2],
                max_attempts=500,
                generate_curves=True)
rhc_run_stats, rhc_run_curves = rhc.run()

# simulated annealing
sa = SARunner(problem,
              experiment_name='sa_n',
              output_directory='out/',
              seed=123,
              iteration_list=2 ** np.arange(14),
              temperature_list=[5],
              max_attempts=5000,
              generate_curves=True)
sa_run_stats, sa_run_curves = sa.run()
# genetic algorithms
ga = GARunner(problem,
              experiment_name='ga_n',
              output_directory='out/',
              seed=123,
              iteration_list=2 ** np.arange(14),
              population_sizes=[5000],
              mutation_rates=[.2]
              )
ga_run_stats, ga_run_curves = ga.run()
# ma = MIMICRunner(problem=problem,
#                  experiment_name='flipflop_n',
#                  output_directory='out/',
#                  seed=123,
#                  iteration_list=2 ** np.arange(14),
#                  max_attempts=5000,
#                  keep_percent_list=[0.20, 0.40, 0.60, 0.80, 1.0],
#                  population_sizes=[200])
# the two data frames will contain the results
