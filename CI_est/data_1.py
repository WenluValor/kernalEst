import numpy as np
import math
import pandas as pd
from scipy import stats

def generate_data(T_value: int, P0_value: int, q_value: int, d_value: int, n_value: int,
                  bound: np.array, p_value: int, b: int):
    np.random.seed(b + 5)

    global T, P0, q, n, d, p
    T = T_value
    P0 = P0_value
    q = q_value
    n = n_value
    d = d_value
    p = p_value

    vec_t = np.zeros([n, d])
    base = round(n**(1 / d))
    sub_vec_t = np.zeros([d, base])
    for i in range(d):
        step = (bound[2 * i + 1] - bound[2 * i]) / base
        sub_vec_t[i, :] = np.arange(bound[2 * i + 1], bound[2 * i], -step)

    for i in range(n):
        ind0 = i % base
        ind1 = int((i % (base ** 2) - ind0) / base)
        ind2 = int((i - ind0 - ind1 * base) / base**2)

        vec_t[i, 0] = sub_vec_t[0, ind0]
        vec_t[i, 1] = sub_vec_t[1, ind1]
        vec_t[i, 2] = sub_vec_t[2, ind2]

    DF = pd.DataFrame(vec_t)
    if b == -1:
        DF.to_csv('0vec_t.csv')
    else:
        DF.to_csv('MSE_outcome/' + str(b) + '_0vec_t.csv')


    for i in range(1, p + 1):
        DF = pd.DataFrame(vec_t)
        if b == -1:
            DF.to_csv(str(i) + 'vec_t.csv')
        else:
            DF.to_csv('MSE_outcome/' + str(b) + '_' + str(i) + 'vec_t.csv')

    Y0 = np.zeros([n])
    for i in range(n):
        Y0[i] = avg_Y0(t=vec_t[i])
    DF = pd.DataFrame(Y0)
    if b == -1:
        DF.to_csv('Y0.csv')
    else:
        DF.to_csv('MSE_outcome/' + str(b) + '_Y0.csv')

    Z = np.zeros([n, p])
    for j in range(p):
        if j == 0:
            for i in range(n):
                t = vec_t[i]
                Z[i, j] = avg_Y1(t)
        elif j == 1:
            for i in range(n):
                t = vec_t[i]
                Z[i, j] = avg_Y2(t)
        else:
            for i in range(n):
                t = vec_t[i]
                Z[i, j] = avg_Y3(t)

    DF = pd.DataFrame(Z)
    if b == -1:
        DF.to_csv('Z.csv')
    else:
        DF.to_csv('MSE_outcome/' + str(b) + '_Z.csv')

def true_f0(t: np.array):
    norm = stats.norm(loc=0, scale=1)
    d1 = 1 / t[2] * (math.log(100 / t[0]) - (t[1] - t[2]**2 / 2))
    return t[0] * norm.cdf(-d1 + t[2]) - 100 * math.exp(-t[1]) * norm.cdf(-d1)

def true_f(t: np.array):
    norm = stats.norm(loc=0, scale=1)
    d1 = 1 / t[:, 2] * (np.log(100 / t[:, 0]) - (t[:, 1] - t[:, 2]**2 / 2))
    return t[:, 0] * norm.cdf(-d1 + t[:, 2]) - 100 * np.exp(-t[:, 1]) * norm.cdf(-d1)

def partial_true_f(t: np.array, part_ind: int):
    # part_ind = 0, 1, ... p-1
    norm = stats.norm(loc=0, scale=1)
    d1 = 1 / t[2] * (math.log(100 / t[0]) - (t[1] - t[2] ** 2 / 2))
    if part_ind == 0:
        return norm.cdf(-d1 + t[2]) + norm.pdf(-d1 + t[2]) / t[2] \
                - 100 * math.exp(-t[1]) / (t[0] * t[2]) * norm.pdf(-d1)
    elif part_ind == 1:
        return t[0] * norm.pdf(-d1 + t[2]) / t[2] + 100 * math.exp(-t[1]) * norm.cdf(-d1)\
                - 100 * math.exp(-t[1]) / t[2] * norm.pdf(-d1)
    else:
        return t[0] * d1 / t[2] * norm.pdf(-d1 + t[2]) - 100 * math.exp(-t[1]) * norm.pdf(-d1)\
                * (d1 / t[2] - 1)

def single_ST(t: np.array, w: float):
    return t[0] * math.exp((t[1] - t[2]**2 / 2) * T + t[2] * math.sqrt(T) * w)

def I(S_T: float):
    if (S_T >= P0):
        return 1
    else: return 0

def avg_Y0(t: np.array):
    ans = np.zeros([q])
    for i in range(q):
        w = np.random.normal(loc=0, scale=1)
        S_T = single_ST(t, w)
        ans[i] = single_Y0(t, S_T)
    return np.mean(ans).item()

def single_Y0(t: np.array, S_T: float):
    return math.exp(-t[1] * T) * max(0, S_T - P0)

def avg_Y1(t: np.array):
    ans = np.zeros([q])
    for i in range(q):
        w = np.random.normal(loc=0, scale=1)
        S_T = single_ST(t, w)
        ans[i] = single_Y1(t, S_T)
    return np.mean(ans).item()

def single_Y1(t: np.array, S_T: float):
    return math.exp(-t[1] * T) * S_T / t[0] * I(S_T)

def avg_Y2(t: np.array):
    ans = np.zeros([q])
    for i in range(q):
        w = np.random.normal(loc=0, scale=1)
        S_T = single_ST(t, w)
        Y_0 = single_Y0(t, S_T)
        ans[i] = single_Y2(t, S_T, Y_0)
    return np.mean(ans).item()

def single_Y2(t: np.array, S_T: float, Y_0: float):
    return -T * Y_0 + math.exp(-t[1] * T) * T * S_T * I(S_T)

def avg_Y3(t: np.array):
    ans = np.zeros([q])
    for i in range(q):
        w = np.random.normal(loc=0, scale=1)
        S_T = single_ST(t, w)
        ans[i] = single_Y3(t, S_T)
    return np.mean(ans).item()

def single_Y3(t: np.array, S_T: float):
    return math.exp(-t[1] * T) / t[2] \
           * (math.log(S_T / t[0]) - (t[1] + t[2]**2 / 2) * T) * S_T * I(S_T)

if __name__ == '__main__':
    exit(0)
