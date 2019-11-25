import mdptoolbox.example
import numpy as np

from plotting import plot_results

forest_VI_csv_list = []
forest_PI_csv_list = []
forest_QL_csv_list = []
GAMMA = [.8, .9, .95, .99]
run_VI = True
run_PI = True
run_QL = True
for gamma in GAMMA:
    if run_VI:
        P, R = mdptoolbox.example.forest(S=10)
        vi = mdptoolbox.mdp.ValueIteration(P, R, gamma, epsilon=.001, set_iter=False)
        vi.setVerbose()
        forest_VI_csv_list.append(vi.run())
        print(vi.V)

    if run_PI:
        P, R = mdptoolbox.example.forest(S=10)
        pi = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
        pi.setVerbose()
        forest_PI_csv_list.append(pi.run())
        #expected = (26.244000000000014, 29.484000000000016, 33.484000000000016)
        #print (all(expected[k] - pi.V[k] < 1e-12 for k in range(len(expected))))
        print (pi.policy)
if run_QL:
    for i in range(1, 5):
        np.random.seed(0)
        P, R = mdptoolbox.example.forest()
        ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
        ql.setVerbose()
        forest_QL_csv_list.append(ql.run(decaytype=i, decayrate=.99))
        # print ("#########")
        # print (ql.Q)
        # print ("#########")
        # expected = (11.198908998901134, 11.741057920409865, 12.259732864170232)
        # print (all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected))))
        #
        # print (ql.policy)
        # print (ql.V)
        # print (ql.time)
        # (0, 1, 1)

plot_results('Forest with 10 states', 'Value Iteration', forest_VI_csv_list, 'Iteration',
             'V_average', 'Iteration', 'Average reward', label_col = 'gamma', show_convergence=True)
plot_results('Forest with 10 states', 'Value Iteration', forest_VI_csv_list, 'Iteration',
             'clock_time', 'Iteration', 'Time', label_col = 'gamma', show_convergence=True)
plot_results('Forest with 10 states', 'Policy Iteration', forest_PI_csv_list, 'Iteration',
             'V_average', 'Iteration', 'Average reward', label_col = 'gamma', show_convergence=True)
plot_results('Forest with 10 states', 'Policy Iteration', forest_PI_csv_list, 'Iteration',
             'clock_time', 'Iteration', 'Time', label_col = 'gamma', show_convergence=True)
plot_results('Forest with 10 states', 'Q-Learning', forest_QL_csv_list, 'episode',
             'V_average', 'Episode', 'Average reward', label_col = 'decay', show_convergence=True)
plot_results('Forest with 10 states', 'Q-Learning', forest_QL_csv_list, 'episode',
             'epsilon', 'Episode', 'Epsilon', label_col = 'decay', show_convergence=True)
plot_results('Forest with 10 states', 'Q-Learning', forest_QL_csv_list, 'episode',
             'time', 'Episode', 'Time', label_col = 'decay', show_convergence=True)