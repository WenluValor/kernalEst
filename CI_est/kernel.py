import math

import numpy as np
import pandas as pd

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.gaussian_process.kernels import Matern

from scipy import stats
import random


class MultiMatern(Matern):
    def __init__(self, l1, l2, l3, nu=2.5):
        super().__init__(length_scale=l1, nu=nu)
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        # self.Matern1 = Matern(length_scale=l1, nu=nu)
        # self.Matern2 = Matern(length_scale=l2, nu=nu)
        # self.Matern3 = Matern(length_scale=l3, nu=nu)

    def __call__(self, X, Y=None, eval_gradient=False):
        index = 0
        x = X[:, index].reshape(-1, 1)
        y = Y[:, index].reshape(-1, 1)

        super().__init__(length_scale=self.l1, nu=self.nu)
        A = super().__call__(x, y)

        index = 1
        x = X[:, index].reshape(-1, 1)
        y = Y[:, index].reshape(-1, 1)
        super().__init__(length_scale=self.l2, nu=self.nu)
        B = super().__call__(x, y)

        index = 2
        x = X[:, index].reshape(-1, 1)
        y = Y[:, index].reshape(-1, 1)
        super().__init__(length_scale=self.l3, nu=self.nu)
        C = super().__call__(x, y)

        return A * B * C

    def get_length(self):
        params = dict()

        params['l1'] = self.l1
        params['l2'] = self.l2
        params['l3'] = self.l3
        return params


def opt_bandwidth():
    X = np.array(pd.read_csv('0vec_t.csv', index_col=0))
    Y = np.array(pd.read_csv('Y0.csv', index_col=0))

    kr = KernelRidge(alpha=1e-3)
    param_dist = {"kernel": [MultiMatern(l1, l2, l3, nu=2.5)
                             for l1 in np.logspace(-3, 3, 15)
                             for l2 in np.logspace(-3, 3, 15)
                             for l3 in np.logspace(-3, 3, 15)]}
    search_cv = RandomizedSearchCV(kr, param_dist, cv=5, n_iter=500, n_jobs=-1)
    search_cv.fit(X, Y)

    best_kernel = search_cv.best_params_['kernel']

    tau = np.zeros([d])
    tau[0] = best_kernel.get_length()['l1']
    tau[1] = best_kernel.get_length()['l2']
    tau[2] = best_kernel.get_length()['l3']

    DF = pd.DataFrame(tau)
    DF.to_csv('tau.csv')
    return


def p(w, tau):
    a = tau / math.sqrt(5)
    ans = 8 * a / (3 * math.pi) / (1 + (a * w) ** 2) ** 3
    return ans


def const_f(tau, gamma):
    a = tau / math.sqrt(5)
    if 1 - 3 * (a * gamma) ** 2 <= 0:
        const = 8 * a * gamma / 3
    else:
        const = 32 / 81 / (a * gamma) / (1 - (a * gamma) ** 2) ** 2
    return const


def sample_p(size: int, tau):
    count = 0
    gamma = 0.1
    cauchy = stats.cauchy(scale=gamma)
    const = const_f(tau, gamma)

    ans = np.zeros([size])

    while count < size:
        x = cauchy.rvs(size=1)
        u = np.random.uniform(0, 1)
        score = p(x, tau) / (const * cauchy.pdf(x))

        if u <= score:
            ans[count] = x
            count += 1
    return ans


def generate_wb(d_value: int, s_value: int):
    random.seed(15)
    np.random.seed(4)

    global d, s
    d = d_value
    s = s_value

    opt_bandwidth()
    tau = np.array(pd.read_csv('tau.csv', index_col=0))

    w_js = np.zeros([d, s])
    b_js = np.zeros([d, s])
    for i in range(d):
        w_js[i] = sample_p(s, tau[i])
        b_js[i] = np.random.uniform(0, 2 * math.pi, size=s)

    DF = pd.DataFrame(w_js)
    DF.to_csv('w_js.csv')
    DF = pd.DataFrame(b_js)
    DF.to_csv('b_js.csv')
    return


if __name__ == '__main__':
    random.seed(15)
    np.random.seed(4)

    generate_wb(d_value=3, s_value=8)


