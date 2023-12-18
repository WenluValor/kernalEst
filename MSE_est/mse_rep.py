import random

import numpy as np
import torch

import kernel as knl
import main
import data_1 as dt_1
import data_2 as dt_2
import data_3 as dt_3
import time

def set_global(p_value, s_value, d_value, B_value, n_value):
    global p, s, d, B, N
    p = p_value
    s = s_value
    d = d_value
    B = B_value
    N = n_value
    return


def generate_data(b: int, case_num: int):
    if case_num == 1:
        dt_1.generate_data(T_value=1, P0_value=100, q_value=5000, d_value=d, n_value=N,
                  bound=np.array([80, 120, 0.01, 0.05, 0.2, 1]), p_value=p, b=b)
        pass
    elif case_num == 2:
        dt_2.generate_data(c_value=np.array([1, 0.8, 0.7, 0.6]), d_value=d, vec_t_up=1.5, vec_t_low=0.5, rho_value=0.4,
                       n_value=N, std_value=0.35, mean_value=0, p_value=p, b=b)
        pass
    elif case_num == 3:
        dt_3.generate_data(p_value=p, n_value=N, d_value=d, up=1, down=0, b=b)
        pass
    return


def regress_mse(case_num: int):
    # Serial computing
    # regression on data, ..., store torch_C
    for b in range(30, B - 1):
        # generate_data(b=b, case_num=case_num)  # generate (b) Y_0, Z, vec_t
        knl.generate_wb(d_value=d, s_value=s, b=b)  # generate (b) w_js, b_js,
        main.create_mat(s=s, d=d, p=p, N=N, b=b)  # generate (0) PSI_mat, part_PSI
        main.boots_compute(b=b, s=s, d=d, p=p, N=N, use_lmd=True)  # generate (b) torch_C, lmd
    return

def special_mse(case_num: int, ind_list: list):
    for b in ind_list:
        # generate_data(b=b, case_num=case_num)  # generate (b) Y_0, Z, vec_t
        # knl.generate_wb(d_value=d, s_value=s, b=b)  # generate (b) w_js, b_js,
        # main.create_mat(s=s, d=d, p=p, N=N, b=b)  # generate (0) PSI_mat, part_PSI
        main.boots_compute(b=b, s=s, d=d, p=p, N=N, use_lmd=False)  # generate (b) torch_C, lmd
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
    set_global(p_value=p, d_value=d, s_value=s, B_value=B, n_value=n)

    b = int(lst[0])
    case_num = int(lst[1])
    generate_data(b=b, case_num=case_num)  # generate (b) Y_0, Z, vec_t
    knl.generate_wb(d_value=d, s_value=s, b=b)  # generate (b) w_js, b_js,
    main.create_mat(s=s, d=d, p=p, b=b, N=N)  # generate (0) PSI_mat, part_PSI
    main.boots_compute(b=b, s=s, d=d, p=p, N=N, use_lmd=True)  # generate (b) torch_C, lmd

if __name__ == '__main__':

    # series computing
    random.seed(15)
    np.random.seed(4)
    torch.manual_seed(10)
    
    s = 5
    n = 21**3
    d = 3
    p = 2
    B = 200
    case_num = 1
    set_global(p_value=p, d_value=d, s_value=s, B_value=B, n_value=n)
    regress_mse(case_num=case_num)
    # ind_list = [129, 140, 145, 176]
    # special_mse(case_num=case_num, ind_list=ind_list)



    '''
    # bash
    case_num = 3
    import sys
    arg1 = sys.argv[1] if len(sys.argv) > 1 else None
    print(arg1)
    lst = (arg1, case_num)
    regress(lst)
    '''



    '''
    # multiprocessing - map
    begin = time.time()
    case_num = 3
    regress((-1, case_num))

    import multiprocessing
    pool = multiprocessing.Pool(processes=4)
    testFL = []
    for i in range(21):
        testFL.append((i, case_num))

    # testFL:要处理的数据列表，run：处理testFL列表中数据的函数
    pool.map(regress, testFL)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()
    end = time.time()
    print(end - begin)
    '''

    '''
    # multiprocessing - apply_async (apply is same as series)
    begin = time.time()
    case_num = 3
    regress((-1, case_num))

    import multiprocessing

    pool = multiprocessing.Pool(processes=4)
    testFL = []
    for i in range(21):
        testFL.append((i, case_num))

    for fn in testFL:
        pool.apply_async(regress, (fn,))
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程的退出
    end = time.time()
    print(end - begin)
    '''

    exit(0)