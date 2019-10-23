RUNNING INSTRUCTION
git repo: https://github.gatech.edu/sbassam3/cs-7641-assignments.git
# 1. install equirements.txt
# 2. python <problem name> <input size>
- valid problem names: tsp, knapsack, fourpeaks, flipflop
- input size is an integer. anything above 10 will take a really long time.
- this will generate graphs under /images
- for tuning, manually run the corresponding function. for example, tweak paramters in tune_flipflop() to your desired
- values, then execute tune_flipflop(). This will generate corresponding values that can be passed to plot_hp_x where x
- is the desired optimization algorithm.
# 3. for NN:
- for hyper parameter tuning use run_nn_gc() in nn_base.py. currently it searches for max_iters only. change to your
- desired parameter inside the function. I used the result of the grid search in the below notebooks
- the code can be found in:
    -- runn_nn_ga.ipynb for Genetic Algorithm
    -- runn_nn_gd.ipynb for Dradient Descent
    -- runn_nn_sa.ipynb for simulated Annealing
    -- runn_nn_rhc.ipynb for random hill climbing






SOURCES:
Hayes, G. (2019). mlrose: Machine Learning, Randomized Optimization and SEarch package for Python.
https://github.com/gkhayes/mlrose. Accessed: 10/2019.
Modified by Andrew Rollings and David S Park more info https://github.com/hiive/mlrose
https://scikit-learn.org/stable/index.html Scikit learn library
