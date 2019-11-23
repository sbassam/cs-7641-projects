import mdptoolbox.example
import forest_mdp
# P, R = mdptoolbox.example.forest(S=10)
# vi = mdptoolbox.mdp.ValueIteration(P, R, 0.96)
# vi.setVerbose()
# vi.run()
# print(vi.V)

P, R = mdptoolbox.example.forest(S=10)
vi = mdptoolbox.mdp.ValueIteration(P, R, 0.95, epsilon=.001, set_iter=False)
vi.setVerbose()
vi.run()
print(vi.V)

P, R = mdptoolbox.example.forest(S=1000)
pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.95)
pi.setVerbose()
pi.run()
expected = (26.244000000000014, 29.484000000000016, 33.484000000000016)
print (all(expected[k] - pi.V[k] < 1e-12 for k in range(len(expected))))

print (pi.policy)


# import numpy as np
# import mdptoolbox, mdptoolbox.example
# np.random.seed(0)
# P, R = mdptoolbox.example.forest()
# ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
# ql.run()
# print ("#########")
# print (ql.Q)
# print ("#########")
# expected = (11.198908998901134, 11.741057920409865, 12.259732864170232)
# print (all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected))))
# 
# print (ql.policy)
# print (ql.V)
# print (ql.time)
# # (0, 1, 1)