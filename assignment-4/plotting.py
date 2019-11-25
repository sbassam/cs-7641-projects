from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
import matplotlib.cm as mplcm


def plot_gw(env, gamma, Vs_VI, pis_VI, algname):
    it = 0
    for (V, pi) in zip(Vs_VI, pis_VI):
        plt.figure(figsize=(4,4))
        plt.imshow(V.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
        ax = plt.gca()
        ax.set_xticks(np.arange(4)-.5)
        ax.set_yticks(np.arange(5)-.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        Y, X = np.mgrid[0:4, 0:4]
        a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
        Pi = pi.reshape(4,4)
        for y in range(4):
            for x in range(4):
                a = Pi[y, x]
                u, v = a2uv[a]
                plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1)
                plt.text(x, y, str(env.unwrapped.desc[y,x].item().decode()),
                         color='g', size=12,  verticalalignment='center',
                         horizontalalignment='center', fontweight='bold')
        plt.grid(color='b', lw=2, ls='-')
        plt.title(algname+": Gridworld at iteration: "+ str(it))
        it += 1
    # plt.figure()
        plt.savefig(
            'images/'+algname+'-gw-nIt'+str(len(Vs_VI))+'-gamma-'+str(gamma)+ str(datetime.now()) + '.png')
        plt.close()

def plot_state_values_vs_iteration(gamma, Vs_VI, algname):
    """https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib"""
    x = range(len(Vs_VI))
    y = Vs_VI
    # fig, ax = plt.subplots()

    cm = plt.get_cmap('gist_rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=len(y[0]) - 1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(len(y[0]))])
    plt.xlabel("ith Iteration")
    plt.ylabel("Value")
    plt.title(algname+": Changes of value over iteration for each state")
    for i in range(len(y[0])):
        ax.plot(x, [pt[i] for pt in y], 'x-', label='state %s' % i)
    if len(x) > 22:
        ax.set_xticks(x[::int(len(x)/20)])
    else:
        ax.set_xticks(x)
    plt.legend()
    plt.savefig(
        'images/'+ algname+'-state-values-vs-iteration-' + str(len(Vs_VI)) + '-gamma-' + str(gamma) + str(datetime.now()) + '.png')
    plt.close()


def plot_results(problem_name, alg_name, csv_list, x_col_name, y_col_name, x_label, y_label, label_col=None,
                 show_convergence=True, logx=False, logy=False, vertical_x=False, vertical_y=False):
    plt.style.use('seaborn')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    num_x_ticks = 0
    for csv in csv_list:
        df = pd.read_csv(csv)
        if label_col:
            label = label_col + ": " + str(df[label_col][0])
        plt.plot(df[x_col_name], df[y_col_name], label=label)
        if logx:
            plt.xscale('log')
        if logy:
            plt.yscale('log')
        if show_convergence:
            plt.plot(df[x_col_name].iat[-1], df[y_col_name].iat[-1], 'g*')
        if df.shape[0] > num_x_ticks:
            num_x_ticks = df.shape[0]
            if num_x_ticks > 22:
                ax.set_xticks(df[x_col_name][::int(num_x_ticks / 20)])
            else:
                ax.set_xticks(df[x_col_name])
        if vertical_x:
            plt.xticks(rotation='vertical')

        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(problem_name + ": " + alg_name + '-' + y_label + ' vs ' + x_label)
    plt.savefig(
        'images/' + problem_name + '-' + alg_name + '--' + y_col_name + '-vs-' + x_col_name + str(datetime.now()) +
        '.png')
    plt.close()



