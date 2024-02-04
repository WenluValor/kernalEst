import os
import random

import numpy as np
import pandas as pd
import torch

import kernel as knl
import main
import data_1 as dt_1
import data_2 as dt_2
import data_3 as dt_3


if __name__ == '__main__':
    random.seed(15)
    np.random.seed(4)
    torch.manual_seed(10)
    s = 5
    n = 7**3
    d = 3
    p = 0
    case_num = 1
    already_exist = True

    # norm_f:
    if already_exist:
        pass
    else:
        if case_num == 1:
            dt_1.generate_data(T_value=1, P0_value=100, q_value=5000, d_value=d, n_value=n,
                               bound=np.array([80, 120, 0.01, 0.05, 0.2, 1]), p_value=p, b=-1)
        elif case_num == 2:
            dt_2.generate_data(c_value=np.array([1, 0.8, 0.7, 0.6]), d_value=d, vec_t_up=1.5, vec_t_low=0.5,
                               rho_value=0.9,
                               n_value=n, std_value=0.35, mean_value=0, p_value=p, b=-1)
        elif case_num == 3:
            dt_3.generate_data(p_value=p, n_value=n, d_value=d, up=1, down=0, b=-1)

        knl.generate_wb(d_value=d, s_value=s)
        main.create_mat(s=s, N=n, d=d, p=p)
    main.boots_compute(N=n, s=s, d=d, p=p, use_lmd=False)




    exit(0)