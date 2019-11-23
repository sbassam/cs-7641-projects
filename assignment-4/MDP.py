import time
import pandas as pd
import numpy as np


class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P
        self.nS = nS
        self.nA = nA
        self.desc = desc

    def value_iteration(mdp, gamma, epsilon, max_iter=1000):
        Vs = [np.zeros(mdp.nS)]
        pis = []
        csv_path = 'out/value-iteration-' + str(mdp.desc.shape) + '-epsilon-' + str(epsilon) + '-gamma-' + str(
            gamma) + '.csv'
        data = []
        cols = ['env_name', 'algname', 'gamma', 'epsilon', 'threshold', 'Iteration', 'variation',
                'V_max_diff', 'clock_time', '# chg action at iter', '# chg actions']
        env_name = 'frozen-lake-' + str(mdp.desc.shape)
        threshold = epsilon * (1 - gamma) / gamma

        t = 0
        nChgActions = 0
        for it in range(max_iter):
            oldpi = pis[-1] if len(pis) > 0 else None
            Vprev = Vs[-1]
            start = time.time()
            V = np.zeros(mdp.nS)
            pi = np.zeros(mdp.nS)
            for state in mdp.P:
                maxv = 0
                for action in mdp.P[state]:
                    v = 0
                    for probability, nextstate, reward in mdp.P[state][action]:
                        v += probability * (reward + gamma * Vprev[nextstate])
                    if v > maxv:
                        maxv = v
                        pi[state] = action
                V[state] = maxv
            end = time.time()
            ####
            t += end - start
            nChgActions += (pi != pi_prev).sum()
            diff = vpi if len(Vs) == 0 else vpi - Vs[-1]
            variation = diff.max() - diff.min()
            max_diff = diff.max()
            row = [env_name, 'PI', gamma, epsilon, threshold, it, variation, max_diff, t, (pi != pi_prev).sum(),
                   nChgActions]
            Vs.append(vpi)
            pis.append(pi)
            pi_prev = pi
            data.append(row)
            if variation < threshold:
                break
                ####
            max_diff = np.abs(V - Vprev).max()
            #nChgActions = "N/A" if oldpi is None else (pi != oldpi).sum()
            Vs.append(V)
            pis.append(pi)
            row = ['value-iteration', it, max_diff, nChgActions, V[0], end-start]
            data.append(row)
        result = pd.DataFrame(data, columns=cols)
        result.to_csv(csv_path, index=None)

        return Vs, pis



    def policy_iteration(mdp, gamma, epsilon, max_iter=1000):
        csv_path = 'out/policy-iteration-'+str(mdp.desc.shape)+'-epsilon-' + str(epsilon) + '-gamma-' + str(gamma) + '.csv'
        data = []
        cols = ['env_name', 'algname', 'gamma', 'epsilon', 'threshold', 'Iteration', 'variation',
                'V_max_diff', 'clock_time', '# chg action at iter', '# chg actions']
        env_name = 'frozen-lake-'+str(mdp.desc.shape)
        threshold = epsilon*(1-gamma)/gamma

        Vs = []
        pis = []
        pi_prev = np.zeros(mdp.nS, dtype='int')
        pis.append(pi_prev)
        t = 0
        nChgActions = 0
        for it in range(max_iter):
            start = time.time()
            vpi = mdp.compute_vpi(pis[-1], gamma=gamma)
            qpi = mdp.compute_qpi(vpi, gamma=gamma)
            pi = qpi.argmax(axis=1)
            end = time.time()
            if it ==0: #skip adding the first iteration to the csv
                Vs.append(vpi)
                pis.append(pi)
                pi_prev = pi
                continue
            t += end-start
            nChgActions += (pi != pi_prev).sum()
            diff = vpi if len(Vs)==0 else vpi - Vs[-1]
            variation = diff.max() - diff.min()
            max_diff = diff.max()
            row = [env_name, 'PI', gamma, epsilon, threshold, it, variation, max_diff, t, (pi != pi_prev).sum(), nChgActions]
            Vs.append(vpi)
            pis.append(pi)
            pi_prev = pi
            data.append(row)
            if variation < threshold:
                break

        result = pd.DataFrame(data, columns=cols)
        result.to_csv(csv_path, index=None)
        return Vs, pis

    def compute_vpi(mdp, pi, gamma):
        a = np.identity(mdp.nS)
        b = np.zeros(mdp.nS)
        for state in range(mdp.nS):
            for probability, next_state, reward in mdp.P[state][pi[state]]:
                a[state][next_state] = a[state][next_state] - gamma*probability
                b[state] += probability * reward
        V = np.linalg.solve(a, b)
        return V

    def compute_qpi(mdp, vpi, gamma):
        Qpi = np.zeros([mdp.nS, mdp.nA])
        for state in range(mdp.nS):
            for action in range(mdp.nA):
                for probability, nextstate, reward in mdp.P[state][action]:
                    Qpi[state][action] += probability * (reward + gamma * vpi[nextstate])
        return Qpi

