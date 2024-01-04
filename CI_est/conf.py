import math
import os
import random

import numpy as np
import pandas as pd
import torch

from main import boots_hat_fn
import main

from scipy import stats

def set_global(p_value, s_value, d_value, B_value):
    global vec_t, Y0, tau, N, Z, C, p, inv_gram, s, d, B
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

    global MAT_PSI, PART_PSI_List
    MAT_PSI = np.array(pd.read_csv('mat_PSI.csv', index_col=0))
    MAT_PSI = torch.from_numpy(MAT_PSI)

    PART_PSI_List = [None] * p
    for i in range(p):
        mat_PSI = np.array(pd.read_csv(str(i) + 'part_mat_PSI.csv', index_col=0))
        PART_PSI_List[i] = torch.from_numpy(mat_PSI)
    return


def get_s_chi(x, b=-1):
    if b == -1:
        Y_star = Y0
        Z_star = Z

        weight = torch.ones([p])
        for i in range(p):
            weight[i] = torch.var(Y0) / torch.var(Z[:, i])

        sig_sqr = torch.ones([1 + p])
        for i in range(1, p + 1):
            sig_sqr[i] = torch.var(Z_star[:, i - 1])
        sig_sqr[0] = torch.var(Y_star)

    else:
        var_star = np.array(pd.read_csv('Boots_outcome/' + str(b) + '_var.csv', index_col=0))[:, 0]
        var_star = torch.from_numpy(var_star)

        weight = torch.ones([p])
        for i in range(p):
            weight[i] = var_star[0] / var_star[i + 1]

        sig_sqr = var_star

    coef = torch.ones([p + 1])
    for i in range(1, p + 1):
        coef[i] = weight[i - 1]
    coef *= 1 / N

    # part_ind = 0, 1, ... p-1
    if os.path.exists('lmd.csv'):
        if b == -1:
            lmd = np.array(pd.read_csv('lmd.csv', index_col=0))[:, 0]
        else:
            lmd = np.array(pd.read_csv('Boots_outcome/' + str(b) + '_lmd.csv', index_col=0))[:, 0]
    else:
        lmd = 1e-9
    lmd = torch.tensor(lmd)

    mat_M = lmd * torch.eye(((s + 1)**d - 1) * (p + 1))
    for i in range(p + 1):
        if i == 0:
            # print( MAT_PSI.shape), ([100, 645])
            tmp = coef[i] * torch.matmul(torch.transpose(MAT_PSI, 0, 1), MAT_PSI)
            mat_M += tmp
        else:
            part_ind = i - 1
            tmp = coef[i] * torch.matmul(torch.transpose(PART_PSI_List[part_ind], 0, 1), PART_PSI_List[part_ind])
            mat_M += tmp

    mat_M = torch.linalg.inv(mat_M).double()

    tmp = coef[0] * torch.sqrt(sig_sqr[0]) * torch.matmul(mat_M, torch.transpose(MAT_PSI, 0, 1))
    mat_sum = torch.matmul(tmp, torch.transpose(tmp, 0, 1))
    for i in range(p):
        tmp = coef[i + 1] * torch.sqrt(sig_sqr[i + 1]) * torch.matmul(mat_M, torch.transpose(PART_PSI_List[i], 0, 1))
        mat_sum += torch.matmul(tmp, torch.transpose(tmp, 0, 1))

    psi_x = main.boots_get_pdPSI(vec_t=x, N=N, s=s, d=d, p=p)
    tmp = torch.matmul(psi_x, mat_sum)
    mat_res = torch.matmul(tmp, torch.transpose(psi_x, 0, 1))
    return torch.sqrt(torch.diag(mat_res)).reshape(-1, 1)


def get_g_star(x):
    # size of N_x * B
    g_star = boots_hat_fn(t=x, b=0, N=N, s=s, d=d, p=p).reshape(-1, 1)
    for b in range(1, B):
        tmp = boots_hat_fn(t=x, b=b, N=N, s=s, d=d, p=p).reshape(-1, 1)
        g_star = torch.cat((g_star, tmp), dim=1)
    return g_star


def get_mat_z(alpha):
    alpha = alpha.detach().numpy()
    norm = stats.norm(loc=0, scale=1)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_alpha = torch.from_numpy(z_alpha)
    mat_z = torch.diag(z_alpha)
    return mat_z


def get_g(x):
    g_0 = boots_hat_fn(t=x, b=-1, N=N, s=s, d=d, p=p).reshape(-1, 1)
    g_0 = torch.kron(g_0, torch.ones([B]))
    return g_0


def get_chi_sigma(x):
    ans = get_s_chi(x=x, b=0)
    for b in range(1, B):
        tmp = get_s_chi(x=x, b=b)
        ans = torch.cat((ans, tmp), dim=1)
    return ans


def get_beta(x, alpha0):
    # alpha0: scalar
    global mat_g, g_star, chi_sigma
    mat_g = get_g(x)
    g_star = get_g_star(x)
    chi_sigma = get_chi_sigma(x)

    tmp = torch.div(g_star - mat_g, chi_sigma)

    q = torch.tensor([alpha0 / 2, 1 - alpha0 / 2]).double()
    res_quan = torch.quantile(tmp, q, dim=1)
    low = torch.min(res_quan[0])
    up = torch.max(res_quan[1])

    return [low, up]


def hat_alpha_xi(base_num: int, alpha0, case_num: int):
    # Warning: only works for d = 3, remain to generalize
    grid_vec_t = np.zeros([base_num ** 3, 3])

    for i in range(base_num ** 3):
        hundred = int(i / base_num ** 2)
        ten = int((i - base_num ** 2 * hundred) / base_num)
        single = i - base_num ** 2 * hundred - base_num * ten
        grid_vec_t[i, 0] = single
        grid_vec_t[i, 1] = ten
        grid_vec_t[i, 2] = hundred

    if base_num == 1:
        pass
    else:
        grid_vec_t /= (base_num - 1)


    if case_num == 1:
        grid_vec_t[:, 0] = grid_vec_t[:, 0] * 40 + 80
        grid_vec_t[:, 1] = grid_vec_t[:, 1] * 0.04 + 0.01
        grid_vec_t[:, 2] = grid_vec_t[:, 2] * 0.8 + 0.2
    elif case_num == 2:
        grid_vec_t += 0.5

    grid_x = torch.from_numpy(grid_vec_t)

    beta_set = get_beta(x=grid_x, alpha0=alpha0)

    return beta_set


def get_final_alpha_xi(alpha0, end: int, case_num: int, start=1):
    round = end - start
    alpha_set = torch.zeros([round, 2])
    for base_num in range(start, end):
        tmp = hat_alpha_xi(base_num, alpha0, case_num=case_num)
        alpha_set[base_num - start, 0] = tmp[0]
        alpha_set[base_num - start, 1] = tmp[1]
        print(base_num)
        print(alpha_set[base_num - start])
    low = torch.min(alpha_set[:, 0]).item()
    up = torch.max(alpha_set[:, 1]).item()
    ans = np.zeros([2])
    ans[0] = low
    ans[1] = up
    return ans


def get_band(x, alpha):
    g_0 = boots_hat_fn(t=x, b=-1, N=N, s=s, d=d, p=p)
    s_chi = get_s_chi(x)[:, 0]
    z_alpha = alpha

    return g_0[:, 0] - s_chi * z_alpha


def plot_band(n_num, alpha, case_num: int):
    x = torch.arange(0, 1, 1 / n_num).reshape(-1, 1)
    cell = torch.arange(0, 1, 1 / n_num).reshape(-1, 1)
    for i in range(d - 1):
        x = torch.cat((x, cell), dim=1)

    if case_num == 1:
        x[:, 0] = x[:, 0] * 40 + 80
        x[:, 1] = x[:, 1] * 0.04 + 0.01
        x[:, 2] = x[:, 2] * 0.8 + 0.2
    elif case_num == 2:
        x = x + 0.5

    up = get_band(x=x, alpha=alpha[0]).numpy()
    low = get_band(x=x, alpha=alpha[1]).numpy()
    mid = boots_hat_fn(t=x, b=-1, N=N, s=s, d=d, p=p)

    import matplotlib.pyplot as plt
    if case_num == 1:
        from data_1 import true_f
    elif case_num == 2:
        from data_2 import true_f
        c = np.array([1, 0.8, 0.7, 0.6])
    else:
        from data_3 import true_f

    x_axis = np.arange(0, 1, 1/n_num)
    y_axis = np.arange(0, 1, 1/n_num)

    tmp = x_axis.reshape(-1, 1)
    tmp = np.repeat(tmp, d, axis=1)
    if case_num == 1:
        tmp[:, 0] = tmp[:, 0] * 40 + 80
        tmp[:, 1] = tmp[:, 1] * 0.04 + 0.01
        tmp[:, 2] = tmp[:, 2] * 0.8 + 0.2
        y_axis = true_f(t=tmp)
    elif case_num == 2:
        tmp = tmp + 0.5
        y_axis = true_f(c=c, t=tmp)
    else:
        for l in range(y_axis.shape[0]):
            y_axis[l] = true_f(t=tmp[l])

    fig, ax = plt.subplots()
    ax.plot(x_axis, low, label='low', linestyle=':', linewidth=1, color='orange', alpha=0.5)
    ax.plot(x_axis, up, label='up', linestyle=':', linewidth=1, color='blue', alpha=0.5)
    ax.plot(x_axis, mid, label='est', color='red', linewidth=0.75, alpha=0.5)
    ax.plot(x_axis, y_axis, label='true', color='green', linewidth=0.75, alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Confidence band for N=%d, p=%d' %(N, p))
    ax.legend(loc="upper right")
    plt.savefig('N=%d p=%d.png' % (N, p), dpi=300)
    plt.show()


def get_probability(alpha, case_num: int):
    if case_num == 1:
        from data_1 import true_f
    elif case_num == 2:
        from data_2 import true_f
        c = np.array([1, 0.8, 0.7, 0.6])
    else:
        from data_3 import true_f

    torch.manual_seed(7)
    size = 10000
    batch = int(size / 10)
    time = 200
    ans = np.zeros([time])
    for i in range(time):
        for j in range(10):
            torch.manual_seed(i * 10 + j)
            x = torch.rand((batch, d))

            if case_num == 1:
                x[:, 0] = x[:, 0] * 40 + 80
                x[:, 1] = x[:, 1] * 0.04 + 0.01
                x[:, 2] = x[:, 2] * 0.8 + 0.2
            elif case_num == 2:
                x = x + 0.5

            up = get_band(x=x, alpha=alpha[0]).numpy()
            low = get_band(x=x, alpha=alpha[1]).numpy()

            x_axis = x.detach().numpy()
            y_axis = np.zeros([x_axis.shape[0]])
            if case_num == 1:
                y_axis = true_f(t=x_axis)
            elif case_num == 2:
                y_axis = true_f(c=c, t=x_axis)
            else:
                for l in range(batch):
                    y_axis[l] = true_f(x_axis[l])

            mask1 = (up - y_axis) > 0
            mask2 = (low - y_axis) < 0
            mat_M = mask1 * mask2
            ans[i] += np.mean(mat_M)
        ans[i] /= 10
        print(ans[i])
        print(str(i) + '------prob-----')

    return np.mean(ans)

def get_area(alpha, case_num: int):
    torch.manual_seed(7)
    size = 10000
    batch = int(size / 10)
    time = 200
    ans = np.zeros([time])

    if case_num == 1:
        vol = 40 * 0.04 * 0.8
    else:
        vol = 1

    for i in range(time):
        for j in range(10):
            torch.manual_seed(i * 100 + j)
            x = torch.rand((batch, d))

            if case_num == 1:
                x[:, 0] = x[:, 0] * 40 + 80
                x[:, 1] = x[:, 1] * 0.04 + 0.01
                x[:, 2] = x[:, 2] * 0.8 + 0.2
            elif case_num == 2:
                x = x + 0.5

            s_chi = get_s_chi(x)[:, 0].numpy()
            z_alpha = alpha[1] - alpha[0]
            tmp = s_chi * z_alpha
            ans[i] += np.mean(tmp)
        ans[i] = ans[i] / 10 * vol
        print(ans[i])
        print(str(i) + '------area-----')
    return np.mean(ans)


if __name__ == '__main__':
    random.seed(15)
    np.random.seed(4)
    torch.manual_seed(10)
    s = 5
    n = 200
    d = 3
    p = 2
    B = 200
    case_num = 2
    alpha = 0.05


    set_global(p_value=p, d_value=d, s_value=s, B_value=B)

    # ans = get_final_alpha_xi(alpha0=alpha, start=1, end=10, case_num=case_num)
    # DF = pd.DataFrame(ans)
    # DF.to_csv('alpha.csv')

    alpha = np.array(pd.read_csv('alpha.csv', index_col=0))[:, 0]

    prob = get_probability(alpha=alpha, case_num=case_num)
    area = get_area(alpha=alpha, case_num=case_num)
    print('--------result-------')
    print(prob)
    print(area)
    plot_band(n_num=1000, alpha=alpha, case_num=case_num)

    exit(0)
