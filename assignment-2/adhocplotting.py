from plotting import plot_fitness_iter, plot_time_iter, plot_fitness_time

ks_csv_path_list = ['out/rhc_Knapsack_n50.csv', 'out/sa_Knapsack_n50.csv', 'out/ga_Knapsack_n50.csv',
                    'out/mimic_Knapsack_n50.csv']

tsp_csv_path_list = ['out/rhc_tsp_n20.csv', 'out/sa_tsp_n20.csv', 'out/ga_tsp_n20.csv',
                     'out/mimic_tsp_n20.csv']

fourpeaks_csv_path_list = ['out/rhc_fourpeaks_n50.csv', 'out/sa_fourpeaks_n50.csv', 'out/ga_fourpeaks_n50.csv',
                           'out/mimic_fourpeaks_n50.csv']
flipflop_csv_path_list = ['out/rhc_flipflop_n200.csv', 'out/sa_flipflop_n200.csv', 'out/ga_flipflop_n200.csv',
                          'out/mimic_flipflop_n200.csv']


# fitness vs iter
# plot_fitness_iter(csv_path_list=ks_csv_path_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='Knapsack',
#                   input_size=50)
# # plot_fitness_iter(csv_path_list=tsp_csv_path_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='TSP',
# #                   input_size=20)
# plot_fitness_iter(csv_path_list=fourpeaks_csv_path_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='Four Peaks',
#                   input_size=50)
# plot_fitness_iter(csv_path_list=flipflop_csv_path_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='Flip Flop',
#                   input_size=200)


# # time vs iter
# plot_time_iter(csv_path_list=ks_csv_path_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='Knapsack',
#                   input_size=50)
# # plot_time_iter(csv_path_list=tsp_csv_path_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='TSP',
# #                   input_size=20)
# plot_time_iter(csv_path_list=fourpeaks_csv_path_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='Four Peaks',
#                   input_size=50)
# plot_time_iter(csv_path_list=flipflop_csv_path_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='Flip Flop',
#                   input_size=200)

# fitness vs time
# plot_fitness_time(csv_path_list=ks_csv_path_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='Knapsack',
#                   input_size=50)
# plot_fitness_time(csv_path_list=tsp_csv_path_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='TSP',
#                   input_size=20)
# plot_fitness_time(csv_path_list=fourpeaks_csv_path_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='Four Peaks',
#                   input_size=50)
# plot_fitness_time(csv_path_list=flipflop_csv_path_list, method_names=['rhc', 'sa', 'ga', 'mimic'], problem_name='Flip Flop',
#                   input_size=200)