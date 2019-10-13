RUNNING INSTRUCTION
# 1. install equirements.txt
# 2. python <problem name> <input size>
- valid problem names: tsp, knapsack, fourpeaks, flipflop
- input size is an integer. anything above 10 will take a really long time.
- this will generate graphs under /images
- for tuning, manually run the corresponding function. for example, tweak paramters in tune_flipflop() to your desired
- values, then execute tune_flipflop(). This will generate corresponding values that can be passed to plot_hp_x where x
- is the desired optimization algorithm.
# 3. for NN:
- the code can be found in nn_base.py. uncomment the corresponding parameters and run run_nn_experiment().
- for the nn graphs run in nn_base.py






SOURCES:
Hayes, G. (2019). mlrose: Machine Learning, Randomized Optimization and SEarch package for Python.
https://github.com/gkhayes/mlrose. Accessed: 10/2019.
Modified by Andrew Rollings and David S Park more info https://github.com/hiive/mlrose
