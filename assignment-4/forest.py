import mdptoolbox.example
import numpy as np

GAMMA = [.8, .9, .95, .99]
run_VI = False
run_PI = False
run_QL = True
for gamma in GAMMA:
    if run_VI:
        P, R = mdptoolbox.example.forest(S=10)
        vi = mdptoolbox.mdp.ValueIteration(P, R, gamma, epsilon=.001, set_iter=False)
        vi.setVerbose()
        vi.run()
        print(vi.V)
    elif run_PI:
        P, R = mdptoolbox.example.forest(S=10)
        pi = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
        pi.setVerbose()
        pi.run()
        #expected = (26.244000000000014, 29.484000000000016, 33.484000000000016)
        #print (all(expected[k] - pi.V[k] < 1e-12 for k in range(len(expected))))
        print (pi.policy)
    elif run_QL:
        for i in range(1, 4):
            np.random.seed(0)
            P, R = mdptoolbox.example.forest()
            ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
            ql.setVerbose()
            ql.run(decaytype=i, decayrate=.99)
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