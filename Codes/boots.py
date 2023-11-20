import random

import numpy as np
import pandas as pd
import torch

from kernel import MultiMatern
import kernel as knl
from main import boots_hat_fn
from main import boots_partial_fn
import main
import data_1 as dt_1
import data_2 as dt_2
import data_3 as dt_3

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
    np.random.seed(b + 1)
    random.seed(b + 1)
    tmp = get_epsilon().detach().numpy()

    rng = np.random.default_rng()
    samples = rng.choice(tmp[:,0], N, replace=True)
    y0 = Y0.detach().numpy()
    Y_star = y0 + samples.reshape(-1, 1)
    DF = pd.DataFrame(Y_star)
    DF.to_csv('outcome/' + str(b) + '_Y0.csv')

    knl.generate_wb(b=b, d_value=d, s_value=s)

    Z_star = np.zeros([N, p])
    for i in range(p):
        tmp = get_partial_epsilon(i).detach().numpy()
        samples = rng.choice(tmp[:,0], N, replace=True)
        z0 = Z.detach().numpy()
        Z_star[:, i] = z0[:, i] + samples

    DF = pd.DataFrame(Z_star)
    DF.to_csv('outcome/' + str(b) + '_Z.csv')
    main.create_mat(s=s, N=N, d=d, p=p, b=b)
    return


def regress_boots():
    # regression on data, ..., store torch_C
    # parellel loop
    for b in range(B):
        generate_data(b) # generate (b) Y_star, Z_star, w_js_star, b_js_star
        main.boots_compute(b=b, N=N, s=s, d=d, p=p)
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


def get_s_chi(x, b=-1, use_lmd=False):
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
    return pi - 1 + alpha0


def get_beta(x, alpha0, left=1e-6, right=1, convergence=1e-6):
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

    # case_num = 1
    '''
    grid_vec_t[:, 0] = grid_vec_t[:, 0] * 40 + 80
    grid_vec_t[:, 1] = grid_vec_t[:, 1] * 0.04 + 0.01
    grid_vec_t[:, 2] = grid_vec_t[:, 2] * 0.8 + 0.2
    '''
    # case_num = 2
    grid_vec_t += 0.5


    grid_x = torch.from_numpy(grid_vec_t)

    beta_set = get_beta(x=grid_x, alpha0=alpha0)

    return torch.quantile(beta_set, xi)


def get_final_alpha_xi(xi, alpha0, end: int, start=1):
    round = end - start
    alpha_set = torch.zeros([round])
    for base_num in range(start, end):
        alpha_set[base_num - start] = hat_alpha_xi(xi, base_num, alpha0)
        print(base_num)
        print(alpha_set[base_num - start])
    return torch.min(alpha_set)


def get_band(x, alpha, is_low=True):
    g_0 = boots_hat_fn(t=x, b=-1, N=N, s=s, d=d, p=p)
    s_chi = get_s_chi(x)
    sigma = torch.sqrt(torch.mean(torch.square(get_epsilon()[: ,0])))
    norm = stats.norm(loc=0, scale=1)
    z_alpha = norm.ppf(1 - alpha / 2)

    if is_low:
        return g_0[:, 0] - s_chi * sigma * z_alpha
    else:
        return g_0[:, 0] + s_chi * sigma * z_alpha

def plot_band(n_num, alpha):
    x = torch.arange(0, 1, 1 / n_num).reshape(-1, 1)
    cell = torch.arange(0, 1, 1 / n_num).reshape(-1, 1)
    for i in range(d - 1):
        x = torch.cat((x, cell), dim=1)

    # case_num = 1
    '''
    x[:, 0] = x[:, 0] * 40 + 80
    x[:, 1] = x[:, 1] * 0.04 + 0.01
    x[:, 2] = x[:, 2] * 0.8 + 0.2
    '''

    # case_num = 2
    x = x + 0.5

    low = get_band(x=x, alpha=alpha, is_low=True).numpy()
    up = get_band(x=x, alpha=alpha, is_low=False).numpy()
    mid = boots_hat_fn(t=x, b=-1, N=N, s=s, d=d, p=p)

    import matplotlib.pyplot as plt
    # case_num = 1
    # from data_1 import true_f
    # case_num = 2
    from data_2 import true_f
    c = np.array([1, 0.8, 0.7, 0.6])

    x_axis = np.arange(0, 1, 1 / n_num)
    y_axis = np.arange(0, 1, 1 / n_num)

    for i in range(x_axis.shape[0]):
        # case_num
        tmp = np.repeat(x_axis[i], d)
        '''
        # case_num = 1
        tmp[0] = tmp[0] * 40 + 80
        tmp[1] = tmp[1] * 0.04 + 0.01
        tmp[2] = tmp[2] * 0.8 + 0.2
        '''
        # case_num = 2
        tmp = tmp + 0.5

        y_axis[i] = true_f(c=c, t=tmp)

    fig, ax = plt.subplots()  # 创建图实例
    ax.plot(x_axis, low, label='low', linestyle=':', linewidth=1, color='orange', alpha=0.5)  # 作y1 = x 图，并标记此线名为linear
    ax.plot(x_axis, up, label='up', linestyle=':', linewidth=1, color='blue', alpha=0.5)  # 作y2 = x^2 图，并标记此线名为quadratic
    ax.plot(x_axis, mid, label='est', color='red', linewidth=0.75, alpha=0.5)  # 作y3 = x^3 图，并标记此线名为cubic
    ax.plot(x_axis, y_axis, label='true', color='green', linewidth=0.75, alpha=0.5)  # 作y3 = x^3 图，并标记此线名为cubic
    ax.set_xlabel('x')  # 设置x轴名称 x label
    ax.set_ylabel('y')  # 设置y轴名称 y label
    ax.set_title('Confidence band for N=%d, p=%d' %(N, p))  # 设置图名为Simple Plot
    ax.legend(loc="upper right")  # 自动检测要在图例中显示的元素，并且显示
    plt.savefig('N=%d p=%d.png' % (N, p), dpi=300)
    plt.show()  # 图形可视化

if __name__ == '__main__':
    random.seed(15)
    np.random.seed(4)
    torch.manual_seed(10)
    s = 5
    n = 500
    d = 3
    p = 2
    B = 200
    case_num = 2

    if case_num == 1:
        dt_1.generate_data(T_value=1, P0_value=100, q_value=5000, d_value=d, n_value=n,
                  bound=np.array([80, 120, 0.01, 0.05, 0.2, 1]), p_value=p)
    elif case_num == 2:
        # dt_2.generate_data(c_value=np.array([1, 0.8, 0.7, 0.6]), d_value=d, vec_t_up=1.5, vec_t_low=0.5, rho_value=0.9,
                       # n_value=n, std_value=0.35, mean_value=0, p_value=p)
        pass
    elif case_num == 3:
        dt_3.generate_data(p_value=p, n_value=n, d_value=d, up=1, down=0)
        pass

    # knl.generate_wb(d_value=d, s_value=s)
    main.create_mat(s=s, N=n, d=d, p=p)
    main.boots_compute(N=n, s=s, d=d, p=p, use_lmd=False)

    # set_global(p_value=p, d_value=d, s_value=s, B_value=B)
    # regress_boots()
    # get_all_sigmasqr()
    # ans = get_final_alpha_xi(xi=0.1, alpha0=0.05, end=6)
    # with open('alpha.csv', 'w') as f:
        # f.write(str(ans.item()))
    # alpha = np.array(pd.read_csv('alpha.csv', header=None))[0]
    # plot_band(n_num=1000, alpha=alpha)

    exit(0)
