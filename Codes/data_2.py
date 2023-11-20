import numpy as np
import math
import pandas as pd


def generate_data(c_value: np.array, d_value: int, vec_t_up: float, vec_t_low: float,
                  rho_value: float, n_value: int, std_value: float, mean_value: float, p_value: int):
    d = d_value
    n = n_value
    c = c_value
    p = p_value

    for i in range(p + 1):
        # vec_t = n * (t_1, t_2, t_4)
        vec_t = np.random.uniform(low=vec_t_low, high=vec_t_up, size=(n, d))
        # vec_t = n * (t_1, t_2, 1, t_4)
        # vec_t = np.insert(vec_t, d - 1, 1, axis=1)
        DF = pd.DataFrame(vec_t)
        name = str(i) + 'vec_t.csv'
        DF.to_csv(name)

    cov = np.eye(d)
    for i in range(d):
        for j in range(d):
            cov[i, j] = rho_value * std_value ** 2
        cov[i, i] = std_value ** 2

    vec_epsilon = np.random.multivariate_normal(mean=np.zeros(d) + mean_value, cov=cov, size=n)
    # epsilon = n * (d + 1)
    vec_epsilon = vec_epsilon[:, 0: p + 1]
    DF = pd.DataFrame(vec_epsilon)
    DF.to_csv('vec_epsilon.csv')

    Y0 = np.zeros([n])
    vec_t = np.array(pd.read_csv('0vec_t.csv', index_col=0))
    vec_t = np.insert(vec_t, d - 1, 1, axis=1)
    for i in range(n):
        Y0[i] = Y_0(eps=vec_epsilon[i, 0], c=c, t=vec_t[i])
    DF = pd.DataFrame(Y0)
    DF.to_csv('Y0.csv')

    Z = np.zeros([n, p])
    for j in range(p):
        vec_t = np.array(pd.read_csv(str(j + 1) + 'vec_t.csv', index_col=0))
        vec_t = np.insert(vec_t, d - 1, 1, axis=1)
        for i in range(n):
            Z[i, j] = partial_Y(eps=vec_epsilon[i, j + 1], c=c, t=vec_t[i], j=j)
    DF = pd.DataFrame(Z)
    DF.to_csv('Z.csv')


def Y_0(eps: float, c: np.array, t: np.array):
    # c: length d+1
    # t: length d+1
    return true_f(c, t) + eps


def partial_Y(eps: float, c: np.array, t: np.array, j: int):
    # j = 0, 1
    return partial_f(c, t, j) + eps


def true_f(c: np.array, t: np.array):
    # c: length d+1
    # t: length d+1
    d1 = t.shape[0]
    d = d1 - 1
    sum_c = sum(c) - c[0] - c[d]
    ans = 1
    for j in range(d1 - 1):
        tmp_c = c[j + 1] / sum_c
        ans *= math.pow(1 / tmp_c, tmp_c) * math.pow(t[j], tmp_c) * math.pow(t[d1 - 1], 1 / sum_c)
    ans *= math.pow(c[0], -1 / sum_c)
    return ans


def partial_f(c: np.array, t: np.array, j: int):
    # partial f / partial t_j
    # j = 0, 1
    d1 = t.shape[0]
    d = d1 - 1
    sum_c = sum(c) - c[0] - c[d]
    ans = true_f(c, t) * 1 / t[j] * c[j + 1] / sum_c
    return ans
