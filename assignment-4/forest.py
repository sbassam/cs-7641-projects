import mdptoolbox.example
P, R = mdptoolbox.example.forest(S=3000)
vi = mdptoolbox.mdp.ValueIteration(P, R, 0.96)
vi.setVerbose()
vi.run()


import numpy as np
import mdptoolbox, mdptoolbox.example
np.random.seed(0)
P, R = mdptoolbox.example.forest()
ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
ql.run()
print ("#########")
print (ql.Q)
print ("#########")
expected = (11.198908998901134, 11.741057920409865, 12.259732864170232)
print (all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected))))

print (ql.policy)
print (ql.V)
print (ql.time)
# (0, 1, 1)