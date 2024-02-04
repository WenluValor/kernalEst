import math
import os
import random

import numpy as np
import pandas as pd
import torch

from main import boots_hat_fn
from main import boots_partial_fn
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

def get_v(x):
    weight = torch.ones([p])
    for i in range(p):
        weight[i] = torch.var(Y0) / torch.var(Z[:, i])

    coef = torch.ones([p + 1])
    for i in range(1, p + 1):
        coef[i] = weight[i - 1]
    coef *= 1 / N

    # part_ind = 0, 1, ... p-1
    if os.path.exists('lmd.csv'):
        lmd = np.array(pd.read_csv('lmd.csv', index_col=0))[:, 0]
    else:
        lmd = 1e-9
        # lmd = 0
    lmd = torch.tensor(lmd)

    # norm_C:
    # mat_M = lmd * torch.eye(((s + 1)**d - 1) * (p + 1))

    # norm_f:
    part_1 = torch.ones([(s + 1)**d - 1])
    part_2 = torch.zeros([((s + 1)**d - 1) * p])
    part_3 = torch.cat((part_1, part_2))
    mat_M = lmd * torch.diag(part_3)
    for i in range(p + 1):
        if i == 0:
            # print( MAT_PSI.shape), ([100, 645]), p=2, n=100
            tmp = coef[i] * torch.matmul(torch.transpose(MAT_PSI, 0, 1), MAT_PSI)
            mat_M += tmp
        else:
            part_ind = i - 1
            tmp = coef[i] * torch.matmul(torch.transpose(PART_PSI_List[part_ind], 0, 1), PART_PSI_List[part_ind])
            mat_M += tmp

    # mat_M = torch.linalg.inv(mat_M) # size: M * M
    mat_M = torch.pinverse(mat_M)

    mat_W = coef[0] * MAT_PSI
    for i in range(p):
        tmp = coef[i + 1] * PART_PSI_List[i]
        mat_W = torch.cat((mat_W, tmp), dim=0)
        # tmp = Z[:, i]
    mat_W = torch.transpose(mat_W, 0, 1) # size: M * ((p+1)n)

    psi_x = main.boots_get_pdPSI(vec_t=x, N=N, s=s, d=d, p=p) # size: size_x * M

    mat_res = psi_x @ mat_M.double() @ mat_W # size: size_x * ((p+1)n)
    return mat_res


def get_epsilon(part_ind=-1):
    if part_ind == -1:
        epsilon = Y0 - boots_hat_fn(t=-1, N=N, s=s, d=d, p=p)
    else:
        epsilon = Z[:, part_ind] - boots_partial_fn(t=-1, part_ind=part_ind, N=N, s=s, d=d, p=p)
    return epsilon


def get_diag_eps():
    eps = get_epsilon()
    for i in range(p):
        tmp = get_epsilon(part_ind=i)
        eps = torch.cat((eps, tmp), dim=0)
    return torch.diag(eps[:, 0])


def get_vx_diag(x):
    vx = get_v(x)
    diag_eps = get_diag_eps()
    return vx @ diag_eps

def get_sigma():
    ans = math.sqrt(N) * torch.norm(vx_diag, p=2, dim=1)
    return ans.reshape(-1, 1)

def band_sigma(x):
    vx = get_v(x)
    diag_eps = get_diag_eps()
    mat = vx @ diag_eps
    return math.sqrt(N) * torch.norm(mat, p=2, dim=1).reshape(-1, 1)

def get_alpha(base_num: int, alpha0, case_num: int):
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

    global vx_diag, sigma
    vx_diag = get_vx_diag(x=grid_x)
    sigma = get_sigma()


    M_set = torch.ones([B])
    for b in range(B):
        torch.manual_seed(b)
        h_b = torch.rand((N, N)).double()
        B_x = vx_diag @ (torch.transpose(h_b, 0, 1) - h_b) @ torch.ones([N]).reshape(-1, 1).double() / math.sqrt(2)
        sigma_inv = 1 / sigma
        tmp = torch.abs(math.sqrt(N) * sigma_inv * B_x)
        M_set[b] = torch.max(tmp)

    return torch.quantile(M_set, alpha0)



def get_band(x, alpha, is_low=True):
    g_0 = boots_hat_fn(t=x, N=N, s=s, d=d, p=p)
    s_chi = band_sigma(x)[:, 0]
    z_alpha = alpha / math.sqrt(N)

    if is_low:
        return g_0[:, 0] - s_chi * z_alpha
    else:
        return g_0[:, 0] + s_chi * z_alpha


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

    low = get_band(x=x, alpha=alpha, is_low=True).numpy()
    up = get_band(x=x, alpha=alpha, is_low=False).numpy()
    mid = boots_hat_fn(t=x, N=N, s=s, d=d, p=p)

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

            low = get_band(x=x, alpha=alpha, is_low=True).numpy()
            up = get_band(x=x, alpha=alpha, is_low=False).numpy()

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

            s_chi = band_sigma(x)[:, 0].numpy()
            z_alpha = alpha / math.sqrt(N)
            tmp = 2 * s_chi * z_alpha
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
    n = 100
    d = 3
    p = 0
    B = 200
    case_num = 2
    alpha = 0.05

    set_global(p_value=p, d_value=d, s_value=s, B_value=B)

    ans = get_alpha(alpha0=alpha, base_num=10, case_num=case_num)

    with open('alpha.csv', 'w') as f:
        f.write(str(ans.item()))
    alpha = np.array(pd.read_csv('alpha.csv', header=None))[0]

    prob = get_probability(alpha=alpha, case_num=case_num)
    area = get_area(alpha=alpha, case_num=case_num)
    print('--------result-------')
    print(prob)
    print(area)
    plot_band(n_num=100, alpha=alpha, case_num=case_num)

    exit(0)
