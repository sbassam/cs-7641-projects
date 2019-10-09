import numpy as np
import pandas as pd
from mlrose import RHCRunner, SARunner, GARunner, MIMICRunner, FlipFlopGenerator

def tune_flipflop():
    input_size = 50
    problem = FlipFlopGenerator.generate(seed=123, size=50)
    # flipflop - RHC
    rhc_tuning = pd.DataFrame()
    
    # run 5 times
    for i in range(5):
        rhc = RHCRunner(problem,
                        experiment_name='rhc_flipflop_n_50_run_' + str(i),
                        output_directory='out/',
                        seed=i,
                        iteration_list=2 ** np.arange(14),
                        restart_list=[1, 5, 10],
                        max_attempts=5000,
                        generate_curves=True)
    
        df_str, rhc_run_curves = rhc.run()
        df_str['run_number'] = i
        rhc_tuning = rhc_tuning.append(df_str)
    rhc_tuning.to_csv(r'out/rhc_flipflop.csv', index=None)
    
    # flipflop - SA
    sa_tuning = pd.DataFrame()
    
    # run 5 times
    for i in range(5):
        sa = SARunner(problem,
                      experiment_name='sa_flipflop_n_50',
                      seed=i,
                      iteration_list=2 ** np.arange(14),
                      temperature_list=[1, 10, 50, 100, 150, 250, 300],
                      max_attempts=5000,
                      generate_curves=True)
        sa_str, sa_run_curves = sa.run()
        sa_str['run_number'] = i
        sa_tuning = sa_tuning.append(sa_str)
    sa_tuning.to_csv(r'out/sa_flipflop.csv', index=None)
    
    # flipflop - GA
    ga_tuning = pd.DataFrame()
    
    # run 5 times
    for i in range(5):
        ga = GARunner(problem,
                      experiment_name='ga_flipflop_n',
                      seed=i,
                      iteration_list=2 ** np.arange(14),
                      population_sizes=[200, 400, 1000, 2500],
                      max_attempts=5000,
                      mutation_rates=[.2, .4],
                      generate_curves=True)
        ga_str, ga_run_curves = ga.run()
        ga_str['run_number'] = i
        ga_tuning = ga_tuning.append(ga_str)
    ga_tuning.to_csv(r'out/ga_flipflop.csv', index=None)
    
    # flipflop MIMIC
    mimic_tuning = pd.DataFrame()
    
    # run 5 times
    for i in range(5):
        mimic = MIMICRunner(problem,
                            experiment_name='mimic_ff_n',
                            seed=i,
                            iteration_list=2 ** np.arange(14),
                            population_sizes=[200, 1000, 3000],
                            max_attempts=5000,
                            keep_percent_list=[.2],
                            generate_curves=True)
        mimic_str, mimic_run_curves = mimic.run()
        mimic_str['run_number'] = i
        mimic_tuning = mimic_tuning.append(mimic_str)
    mimic_tuning.to_csv(r'out/mimic_flipflop.csv', index=None)
    
    
def run_rhc_flipflop(restart, max_attempts=5000, iters_exp=14, num_runs=5, input_size=50):
    problem = FlipFlopGenerator.generate(seed=123, number_of_items_types=input_size)
    rhc_stats = pd.DataFrame()
    rhc_path = 'out/rhc_flipflop_n' + str(input_size) + '.csv'

    # run num_runs times
    for i in range(num_runs):
        rhc = RHCRunner(problem,
                        experiment_name='rhc_run_' + str(i),
                        output_directory='out/',
                        seed=i,
                        iteration_list=2 ** np.arange(iters_exp),
                        restart_list=[restart],
                        max_attempts=max_attempts,
                        generate_curves=True)

        df_stats, rhc_run_curves = rhc.run()
        df_stats['run_number'] = i
        rhc_stats = rhc_stats.append(df_stats)
    rhc_stats.to_csv(rhc_path, index=None)
    return rhc_path


def run_sa_flipflop(temperature, max_attempts=5000, iters_exp=14, num_runs=5, input_size=50):
    problem = FlipFlopGenerator.generate(seed=123, number_of_items_types=input_size)
    sa_stats = pd.DataFrame()
    sa_path = 'out/sa_flipflop_n' + str(input_size) + '.csv'

    # run num_runs times
    for i in range(num_runs):
        sa = SARunner(problem,
                      experiment_name='sa_flipflop_n_50',
                      seed=i,
                      iteration_list=2 ** np.arange(iters_exp),
                      temperature_list=[temperature],
                      max_attempts=max_attempts,
                      generate_curves=True)
        df_stats, sa_run_curves = sa.run()
        df_stats['run_number'] = i
        sa_stats = sa_stats.append(df_stats)
    sa_stats.to_csv(sa_path, index=None)
    return sa_path


def run_ga_flipflop(mut_rate, pop_size, max_attempts=5000, iters_exp=14, num_runs=5, input_size=50):
    problem = FlipFlopGenerator.generate(seed=123, number_of_items_types=input_size)
    ga_stats = pd.DataFrame()
    ga_path = 'out/ga_flipflop_n' + str(input_size) + '.csv'

    # run num_runs times
    for i in range(num_runs):
        ga = GARunner(problem,
                      experiment_name='ga_flipflop_n',
                      seed=i,
                      iteration_list=2 ** np.arange(iters_exp),
                      population_sizes=[pop_size],
                      max_attempts=max_attempts,
                      mutation_rates=[mut_rate],
                      generate_curves=True)
        df_stats, ga_run_curves = ga.run()
        df_stats['run_number'] = i
        ga_stats = ga_stats.append(df_stats)
    ga_stats.to_csv(ga_path, index=None)
    return ga_path


def run_mimic_flipflop(keep_pct, pop_size, max_attempts=5000, iters_exp=14, num_runs=1, input_size=50):
    problem = FlipFlopGenerator.generate(seed=123, number_of_items_types=input_size)
    mimic_stats = pd.DataFrame()
    mimic_path = 'out/mimic_flipflop_n' + str(input_size) + '.csv'

    # run num_runs times
    for i in range(num_runs):
        mimic = MIMICRunner(problem,
                            experiment_name='mimic_flipflop_n',
                            seed=i,
                            iteration_list=2 ** np.arange(iters_exp),
                            population_sizes=[pop_size],
                            max_attempts=max_attempts,
                            keep_percent_list=[keep_pct],
                            generate_curves=True)
        df_stats, mimic_run_curves = mimic.run()
        df_stats['run_number'] = i
        mimic_stats = mimic_stats.append(df_stats)
    mimic_stats.to_csv(mimic_path, index=None)
    return mimic_path
