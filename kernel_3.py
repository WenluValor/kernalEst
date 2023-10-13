import math
import pandas as pd
import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar
from cmath import exp

def opt_bandwith(index: int, p_value, low, up, fold=5):
    # index = 0, 1, 2, dimension
    minimum = minimize_scalar(func, bounds=(0, 1), method='bounded', args=(index, p_value, low, up, fold))
    return minimum.x

def func(bd, index: int, p_value, low, up, fold):
    vec_t = np.array([0, 0, 0])

    for i in range(p_value + 1):
        file_name = str(i) + 'vec_t.csv'
        tmp_t = np.array(pd.read_csv(file_name, index_col=0))
        vec_t = np.vstack((vec_t, tmp_t))
    vec_t = np.delete(vec_t, 0, axis=0)

    t = vec_t[:, index]
    n = t.shape[0]
    ans = 0
    for k in range(fold):
        ind_set = range(k, n, fold)
        new_data = t[ind_set]
        old_data = np.delete(t, ind_set)
        ans += est_K(old_data, new_data, bd, low, up)
    ans /= fold
    return -ans

def est_K(old_data, new_data, bd, low, up):
    # for each data in new, sum their log density by old
    ans = 0
    for x in new_data:
        ans += log_likelihood_K(old_data, x, bd, low, up)
    return ans

def log_likelihood_K(data, x, bd, low, up):
    # return log lkelihood of density with bandwidth bd, at the testing points delta (which has been scaled by bd)
    ans = 0
    scale, err = integrate.quad(single_K, low, up, args=(bd))

    for dt in data:
        delta = abs(x - dt)
        ans += single_K(delta, bd)
    ans /= (data.shape[0]) * scale
    return math.log(ans)

def single_K(t, tau):
    # t is the distance
    delta = abs(t) / tau
    return (1 + delta + delta**2 / 3) * math.exp(-delta)

def int_f(t, w, low, up, tau):
    # Flourier transformation for one dim
    ans = single_K(t, tau) * exp(2 * math.pi * w * t * -1j).real
    return ans

def unscaled_p(w, low, up, tau):
    # low = -1
    # up = 1
    # for one dimension
    v, err = integrate.quad(int_f, low, up, args=(w, low, up, tau))
    return v

def p(w, low, up, tau):
    # low = -1
    # up = 1
    # for one dimension
    sum = 0
    for i in range(10):
        scale, err = integrate.quad(unscaled_p, i * 10, (i + 1) * 10, args=(low, up, tau))
        sum += scale
    sum *= 2 # integral is symmetric
    return unscaled_p(w, low, up, tau) / sum

def const_f(w, low, up, tau):
    # reject sample from U(-5, 5)
    return -p(w, low, up, tau) / (1 / 10)

def search_const(low, up, tau):
    res = minimize_scalar(const_f, args=(low, up, tau))
    const = -res.fun
    return const

def sample_p(size: int, low, up, tau):
    count = 0
    const = search_const(low, up, tau)
    # print(const)
    ans = np.zeros([size])
    while (count < size):
        x = np.random.uniform(-5, 5)
        u = np.random.uniform(0, 1)
        score = p(x, low, up, tau) / (const / 10)
        if (u <= score):
            ans[count] = x
            count += 1
    return ans

def generate_wb(d: int, s: int, low, up, p_value):
    # d = 3
    # w, b: d*s
    w = np.zeros([d, s])
    b = np.zeros([d, s])
    tau = np.zeros([d])
    for i in range(d):
        tau[i] = opt_bandwith(i, p_value, low, up)

    for i in range(d):
        w[i] = sample_p(s, low, up, tau[i])
        b[i] = np.random.uniform(0, 2 * math.pi, size=s)

    DF = pd.DataFrame(w)
    DF.to_csv('w_js.csv')
    DF = pd.DataFrame(b)
    DF.to_csv('b_js.csv')
