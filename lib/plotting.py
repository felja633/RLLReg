import matplotlib.pyplot as plt
from lib.utils import compute_angular_error
import numpy as np
import torch
from lib.tensorlist import TensorListList

def plot_recall_curves(errors, filename, max_err=4.0, fignum=1, key="R_err", type=None, axvline=None, order=None, unit='cm'):
    # errors: dict with errors for each method
    if type is None:
        type=key

    def compute_recall_curves(errors, max_ang, type):

        if type =="R_err":
            out_err = compute_angular_error(errors)
        elif type == "t_err":
            out_err = errors
        elif type == "ang_err":
            out_err = errors
        else:
            out_err = errors
            print("unknown error type")

        out_err = out_err.numpy()
        N = float(out_err.size)
        r = []
        for xi in max_ang:
            M = float((out_err < xi).sum())
            r.append(M / N)

        return r

    x = np.arange(0, max_err, 1./100.)
    names = []
    fig = plt.figure(fignum)

    colors = ['#000000', '#3B7A57', '#9966CC', '#00FFFF', '#007FFF', '#FFBF00', '#848482',
            '#E52B50', '#7FFF00', '#FE6F5E', '#DEB887']

    if order is None:
        for m, c in zip(errors, colors):
            r = compute_recall_curves(errors[m][key], x, type=type)
            plt.plot(x, r, color=c)
            names.append(m)
    else:
        for i, m in enumerate(order):
            r = compute_recall_curves(errors[m][key], x, type=type)
            plt.plot(x, r, color=colors[i])
            names.append(m)

    if not axvline is None:
        plt.axvline(x=axvline, color='k', linestyle='--')

    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, max_err)
    xlabel = 'Threshold (' + unit + ')'
    plt.xlabel(xlabel)
    plt.ylabel('Recall')
    plt.legend(names, loc='lower right')
    plt.savefig(filename)
    plt.show()
    plt.close(fig)