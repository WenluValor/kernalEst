import os
import random

import numpy as np
import pandas as pd
import torch

import kernel as knl
from main import boots_hat_fn
from main import boots_partial_fn
import main
import data_1 as dt_1
import data_2 as dt_2
import data_3 as dt_3

from scipy import stats

def set_global(p_value, s_value, d_value, B_value):
    global vec_t, Y0, tau, N, Z, C, p, s, d, B
    vec_t = np.array(pd.read_csv('0vec_t.csv', index_col=0))
    vec_t = torch.from_numpy(vec_t)

    Y0 = np.array(pd.read_csv('Y0.csv', index_col=0))
    Y0 = torch.from_numpy(Y0)

    tau = np.array(pd.read_csv('tau.csv', index_col=0))
    tau = torch.from_numpy(tau)

    Z = np.array(pd.read_csv('Z.csv', index_col=0))
    Z = torch.from_numpy(Z)

    C = np.array(pd.read_csv('torch_C.csv', index_col=0))[:, 0]
    C = torch.from_numpy(C).float()

    N = vec_t.shape[0]
    p = p_value
    s = s_value
    d = d_value
    B = B_value

    return


def get_epsilon():
    epsilon = Y0 - boots_hat_fn(t=-1, b=-1, N=N, s=s, d=d, p=p)
    epsilon -= torch.mean(epsilon)
    return epsilon


def get_partial_epsilon(index):
    epsilon = Z[:, index] - boots_partial_fn(t=-1, part_ind=index,  b=-1, N=N, s=s, d=d, p=p)
    epsilon -= torch.mean(epsilon)
    return epsilon


def generate_data(b: int):
    np.random.seed(b + 1)

    tmp = get_epsilon().detach().numpy()
    rng = np.random.default_rng()
    samples = rng.choice(tmp[:,0], N, replace=True)
    y0 = boots_hat_fn(t=-1, b=-1, N=N, s=s, d=d, p=p).detach().numpy()
    Y_star = y0 + samples.reshape(-1, 1)
    DF = pd.DataFrame(Y_star)
    DF.to_csv('Boots_outcome/' + str(b) + '_Y0.csv')

    Z_star = np.zeros([N, p])
    for i in range(p):
        tmp = get_partial_epsilon(i).detach().numpy()
        samples = rng.choice(tmp[:,0], N, replace=True)
        z0 = boots_partial_fn(t=-1, part_ind=i,  b=-1, N=N, s=s, d=d, p=p).detach().numpy()[:, 0]
        Z_star[:, i] = z0 + samples

    DF = pd.DataFrame(Z_star)
    DF.to_csv('Boots_outcome/' + str(b) + '_Z.csv')
    return


def regress_boots():
    # regression on data, ..., store torch_C
    for b in range(B):
        print(b)
        generate_data(b) # generate (b) Y_0, Z
        main.boots_compute(b=b, N=N, s=s, d=d, p=p, use_lmd=True) # generate (b) torch_C
    return


def regress(lst):
    # Parallel computing
    random.seed(15)
    np.random.seed(4)
    torch.manual_seed(10)

    s = 5
    n = 100
    d = 3
    p = 2
    B = 5

    b = int(lst[0])
    case_num = int(lst[1])
    if b == -1:
        if case_num == 1:
            dt_1.generate_data(T_value=1, P0_value=100, q_value=5000, d_value=d, n_value=n,
                  bound=np.array([80, 120, 0.01, 0.05, 0.2, 1]), p_value=p)
        elif case_num == 2:
            dt_2.generate_data(c_value=np.array([1, 0.8, 0.7, 0.6]), d_value=d, vec_t_up=1.5, vec_t_low=0.5, rho_value=0.9,
                       n_value=n, std_value=0.35, mean_value=0, p_value=p)
        elif case_num == 3:
            dt_3.generate_data(p_value=p, n_value=n, d_value=d, up=1, down=0)

        knl.generate_wb(d_value=d, s_value=s)
        main.create_mat(s=s, N=n, d=d, p=p)
        main.boots_compute(N=n, s=s, d=d, p=p, use_lmd=True)
    else:
        set_global(p_value=p, d_value=d, s_value=s, B_value=B)
        generate_data(b)  # generate (b) Y_0, Z
        main.boots_compute(b=b, N=N, s=s, d=d, p=p, use_lmd=True)  # generate (b) torch_C

    return


if __name__ == '__main__':
    # seires computing

    random.seed(15)
    np.random.seed(4)
    torch.manual_seed(10)
    s = 5
    n = 200
    d = 3
    p = 2
    B = 200
    case_num = 2
    already_exist = True

    if already_exist:
        pass
    else:
        if case_num == 1:
            dt_1.generate_data(T_value=1, P0_value=100, q_value=5000, d_value=d, n_value=n,
                  bound=np.array([80, 120, 0.01, 0.05, 0.2, 1]), p_value=p)
        elif case_num == 2:
            dt_2.generate_data(c_value=np.array([1, 0.8, 0.7, 0.6]), d_value=d, vec_t_up=1.5, vec_t_low=0.5, rho_value=0.9,
                       n_value=n, std_value=0.35, mean_value=0, p_value=p)
        elif case_num == 3:
            dt_3.generate_data(p_value=p, n_value=n, d_value=d, up=1, down=0)

        knl.generate_wb(d_value=d, s_value=s)
        main.create_mat(s=s, N=n, d=d, p=p)
        main.boots_compute(N=n, s=s, d=d, p=p, use_lmd=False)

    set_global(p_value=p, d_value=d, s_value=s, B_value=B)
    regress_boots()

    # bash
    '''
    case_num = 2
    import sys

    arg1 = sys.argv[1] if len(sys.argv) > 1 else None
    print(arg1)
    lst = (arg1, case_num)
    regress(lst)
    '''


    exit(0)