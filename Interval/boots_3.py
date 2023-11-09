import random

import numpy as np
import pandas as pd
import torch

from kernel_3 import MultiMatern
import kernel_3 as knl_3
from main_3 import boots_hat_fn
from main_3 import boots_partial_fn
import main_3

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

    lmd = 1e-3
    multi_k = MultiMatern(l1=tau[0].item(), l2=tau[1].item(), l3=tau[2].item())

    gram_mat = multi_k.__call__(vec_t.numpy(), vec_t.numpy())
    gram_mat = torch.from_numpy(gram_mat)

    inv_gram = torch.linalg.inv(gram_mat + lmd * torch.eye(N))
    return


def get_epsilon(b=-1):
    epsilon = Y0 - boots_hat_fn(t=b, b=b, N=N, s=s, d=d, p=p)
    epsilon -= torch.mean(epsilon)
    return epsilon


def get_partial_epsilon(index, b=-1):
    # part_t = np.array(pd.read_csv(str(index + 1) + 'vec_t.csv', index_col=0))
    # part_t = torch.from_numpy(part_t)

    epsilon = Z[:, index] - boots_partial_fn(t=b, part_ind=index,  b=b, N=N, s=s, d=d, p=p)
    epsilon -= torch.mean(epsilon)
    return epsilon


def generate_data(b: int):
    tmp = get_epsilon().detach().numpy()
    samples = np.random.choice(tmp[0], N, replace=True)
    y0 = Y0.detach().numpy()
    Y_star = y0 + samples.reshape(-1, 1)
    DF = pd.DataFrame(Y_star)
    DF.to_csv('outcome/' + str(b) + '_Y0.csv')

    knl_3.generate_wb(b=b, d_value=d, s_value=s)
    main_3.create_mat(b)

    Z_star = Z.detach().numpy()
    for i in range(p):
        tmp = get_partial_epsilon(i).detach().numpy()
        samples = np.random.choice(tmp[0], N, replace=True)
        Z_star[:, i] = Z[:, i].detach().numpy() + samples
    DF = pd.DataFrame(Z_star)
    DF.to_csv('outcome/' + str(b) + '_Z.csv')
    return


def regress_boots():
    # regression on data, ..., store torch_C
    # parellel loop
    for b in range(B):
        generate_data(b) # generate (b) Y_star, Z_star, w_js_star, b_js_star
        main_3.boots_compute(b=b, N=N, s=s, d=d, p=p)
    return


def get_sigma():
    eps_set = get_epsilon(0).reshape(1, -1) # shape of N * B
    for b in range(1, B):
        tmp = get_epsilon(b).reshape(1, -1)
        eps_set = torch.cat((eps_set, tmp), dim=0)
    # shape of 1 * B
    return torch.sqrt(torch.mean(torch.square(eps_set), dim=1))


def get_all_sigmasqr():
    global SigmaSqr
    SigmaSqr = get_sigma()
    return


def get_s_chi(x):
    multi_k = MultiMatern(l1=tau[0].item(), l2=tau[1].item(), l3=tau[2].item())
    small_k = multi_k.__call__(vec_t, x)
    small_k = torch.from_numpy(small_k)

    tmp = torch.matmul(inv_gram, small_k)
    return torch.sqrt(torch.sum(torch.square(tmp), dim=0))


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

def get_pi(alpha, x):
    mat_z = get_mat_z(alpha)
    mat_g = get_g(x)
    g_star = get_g_star(x)
    s_chi = get_s_chi(x).reshape(-1, 1)
    sigma = SigmaSqr
    chi_sigma = torch.kron(s_chi, sigma)


    mat_A = torch.matmul(mat_z, chi_sigma)
    mat_L = g_star - mat_A
    mat_U = g_star + mat_A

    mask1 = (mat_U - mat_g) > 0
    mask2 = (mat_L - mat_g) < 0
    mat_M = mask1 * mask2

    vec_pi = torch.mean(mat_M.float(), dim=1) # of shape (N_x)
    return vec_pi


def calcu_pi(alpha, x, alpha0):
    pi = get_pi(alpha, x)
    return pi - alpha0


def get_beta(x, alpha0, left=0, right=1, convergence=1e-6):
    # alpha0: scalar
    size = x.shape[0]
    alpha_left = torch.ones([size]) * left
    alpha_right = torch.ones([size]) * right
    vec_error = torch.ones([size]) * (convergence + 1)
    vec_convrg = torch.ones([size]) * convergence
    cur_root = alpha_left
    count = 0
    while (torch.all((vec_error - vec_convrg) > 0)) & (count < 20):
        f_left = calcu_pi(alpha_left, x, alpha0)
        f_right = calcu_pi(alpha_right, x, alpha0)
        if torch.all((torch.abs(f_left) - vec_convrg) < 0):
            cur_root = alpha_left
        elif torch.all((torch.abs(f_right) - vec_convrg) < 0):
            cur_root = alpha_right
        else:
            mid = (alpha_left + alpha_right) / 2
            f_mid = calcu_pi(mid, x, alpha0)
            f_mask = (f_mid * f_left) < 0
            # alpha right = mid when f_mask True, and remains the same in False places
            # alpha_left - mid when f_mask False, and remains the same in True places
            alpha_right = alpha_right * torch.logical_not(f_mask) + mid * f_mask
            alpha_left = alpha_left * f_mask + mid * torch.logical_not(f_mask)
            cur_root = alpha_left
        vec_error = abs(calcu_pi(cur_root, x, alpha0))
        count += 1

    '''
    # for one scalae
    error = convergence + 1
    cur_root = left
    count = 0
    while (error > convergence) & (count < 20):
        if abs(calcu_pi(left, x, alpha0, B)) < convergence:
            cur_root = left
        elif abs(calcu_pi(right, x, alpha0, B)) < convergence:
            cur_root = right
        else:
            middle = (left + right) / 2
            if (calcu_pi(left, x, alpha0, B) * calcu_pi(middle, x, alpha0, B)) < 0:
                right = middle
            else:
                left = middle
            cur_root = left
        error = abs(calcu_pi(cur_root, x, alpha0, B))
        count += 1
    return cur_root
    '''
    return cur_root


def hat_alpha_xi(xi, base_num: int, alpha0):
    grid_vec_t = np.zeros([base_num ** 3, 3])

    for i in range(base_num ** 3):
        hundred = int(i / base_num ** 2)
        ten = int((i - base_num ** 2 * hundred) / base_num)
        single = i - base_num ** 2 * hundred - base_num * ten
        grid_vec_t[i, 0] = single
        grid_vec_t[i, 1] = ten
        grid_vec_t[i, 2] = hundred
    grid_vec_t /= (base_num ** 3)

    grid_x = torch.from_numpy(grid_vec_t)

    beta_set = get_beta(x=grid_x, alpha0=alpha0)

    return torch.quantile(beta_set, xi)


def get_final_alpha_xi(xi, alpha0, end: int, start=1):
    round = end - start
    alpha_set = torch.zeros([round])
    for base_num in range(start, end):
        alpha_set[base_num - start] = hat_alpha_xi(xi, base_num, alpha0)
    return torch.min(alpha_set)


def get_band(x, alpha, is_low=True):
    g_0 = boots_hat_fn(t=x, b=-1, N=N, s=s, d=d, p=p)
    s_chi = get_s_chi(x)
    sigma = torch.mean(get_epsilon())
    norm = stats.norm(loc=0, scale=1)
    z_alpha = norm.ppf(1 - alpha / 2)

    if is_low:
        return g_0 - s_chi * sigma * z_alpha
    else:
        return g_0 + s_chi * sigma * z_alpha


if __name__ == '__main__':
    random.seed(15)
    np.random.seed(4)
    torch.manual_seed(10)
    set_global(p_value=3, d_value=3, s_value=8, B_value=6)
    regress_boots()
    get_all_sigmasqr()
    ans = get_final_alpha_xi(xi=0.1, alpha0=0.01, end=5)
    with open('alpha.csv', 'w') as f:
        f.write(str(ans.item()))

    exit(0)