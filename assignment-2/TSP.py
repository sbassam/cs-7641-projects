import sklearn
import pandas as pd
import numpy as np
from mlrose import RHCRunner, SARunner, GARunner, MIMICRunner, TSPGenerator

from mlrose import RHCRunner, SARunner, GARunner, MIMICRunner, TSPGenerator

problem = TSPGenerator.generate(seed=123, number_of_cities=10)
rhc_tuning = pd.DataFrame()

## run 5 times
for i in range(5):
    rhc = RHCRunner(problem,
                    experiment_name='rhc_tsp_n_10_run_' + str(i),
                    output_directory='out/',
                    seed=i,
                    iteration_list=2 ** np.arange(16),
                    restart_list=[1, 5, 10, 20, 50, 100],
                    max_attempts=5000,
                    generate_curves=True)

    df_str, rhc_run_curves = rhc.run()
    df_str['run_number'] = i
    rhc_tuning = rhc_tuning.append(df_str)