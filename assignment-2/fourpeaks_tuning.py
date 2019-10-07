import numpy as np
import pandas as pd
from mlrose import FourPeaks, RHCRunner, SARunner, GARunner, MIMICRunner, DiscreteOpt

fitness = FourPeaks()
problem = DiscreteOpt(length=50, fitness_fn=fitness, maximize=True, max_val=2)

# FourPeaks - RHC
rhc_tuning = pd.DataFrame()

# run 5 times
for i in range(5):
    rhc = RHCRunner(problem,
                    experiment_name='rhc_FourPeaks_n_50_run_' + str(i),
                    output_directory='out/',
                    seed=i,
                    iteration_list=2 ** np.arange(14),
                    restart_list=[1, 5, 10],
                    max_attempts=5000,
                    generate_curves=True)

    df_str, rhc_run_curves = rhc.run()
    df_str['run_number'] = i
    rhc_tuning = rhc_tuning.append(df_str)
rhc_tuning.to_csv(r'out/rhc_FourPeaks.csv', index=None)

# FourPeaks - SA
sa_tuning = pd.DataFrame()

# run 5 times
for i in range(5):
    sa = SARunner(problem,
                  experiment_name='sa_FourPeaks_n_50',
                  seed=i,
                  iteration_list=2 ** np.arange(14),
                  temperature_list=[1, 10, 50, 100, 150, 250, 300],
                  max_attempts=5000,
                  generate_curves=True)
    sa_str, sa_run_curves = sa.run()
    sa_str['run_number'] = i
    sa_tuning = sa_tuning.append(sa_str)
sa_tuning.to_csv(r'out/sa_FourPeaks.csv', index=None)

# FourPeaks - GA
ga_tuning = pd.DataFrame()

# run 5 times
for i in range(5):
    ga = GARunner(problem,
                  experiment_name='ga_FourPeaks_n',
                  seed=i,
                  iteration_list=2 ** np.arange(14),
                  population_sizes=[200, 400, 1000, 2500],
                  max_attempts=5000,
                  mutation_rates=[.2, .4],
                  generate_curves=True)
    ga_str, ga_run_curves = ga.run()
    ga_str['run_number'] = i
    ga_tuning = ga_tuning.append(ga_str)
ga_tuning.to_csv(r'out/ga_FourPeaks.csv', index=None)

# FourPeaks MIMIC
mimic_tuning = pd.DataFrame()

# run 5 times
for i in range(5):
    mimic = MIMICRunner(problem,
                        experiment_name='mimic_FourPeaks_n',
                        seed=i,
                        iteration_list=2 ** np.arange(14),
                        population_sizes=[200, 1000, 3000],
                        max_attempts=5000,
                        keep_percent_list=[.2],
                        generate_curves=True)
    mimic_str, mimic_run_curves = mimic.run()
    mimic_str['run_number'] = i
    mimic_tuning = mimic_tuning.append(mimic_str)
mimic_tuning.to_csv(r'out/mimic_FourPeaks.csv', index=None)
