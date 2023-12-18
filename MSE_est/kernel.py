import math

import numpy as np
import pandas as pd

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import Matern

from scipy import stats
from sklearn.model_selection import KFold
import gpytorch
import torch
import time
import os


class MultiMatern(Matern):
    def __init__(self, l1, l2, l3, nu=2.5):
        super().__init__(length_scale=l1, nu=nu)
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

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


def tau_func(tau, fold_num=5, lmd=1e-3):
    kf = KFold(n_splits=fold_num, random_state=0, shuffle=True)
    torch_X = torch.from_numpy(X)
    torch_Y = torch.from_numpy(Y)
    sum = torch.tensor([0])
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        covar_module = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d, lengthscale=torch.abs(tau))
        kernel_matrix = covar_module(torch_X[train_index]).to_dense()
        kernel_matrix = torch.linalg.inv(kernel_matrix + lmd * torch.eye(kernel_matrix.shape[0]))
        kernel_vec = covar_module(torch_X[test_index], torch_X[train_index]).to_dense()
        Y_pred = torch.matmul(torch.matmul(kernel_vec, kernel_matrix), torch_Y[train_index])
        sum = sum + torch.mean(torch.square(Y_pred - torch_Y[test_index]))
    return sum

def tau_descend(initX, tol, lr):
    optimizer = torch.optim.Adam([initX], lr=lr)
    updateX = initX * 2 - 1
    loss = tau_func(updateX)
    last_loss = 3 * loss.item()
    new_loss = 2 * loss.item()
    count = 0

    while (last_loss - new_loss > tol) | (last_loss < new_loss):
        last_loss = new_loss

        updateX = initX
        loss = tau_func(updateX)
        new_loss = loss.item()
        lossABackward = torch.sum(loss)
        optimizer.zero_grad()
        lossABackward.backward()
        optimizer.step()

        # print(last_loss)
        # print(new_loss)
        # print(count)
        # print('-----------tau=========')
        count += 1

    updateX = initX
    loss = tau_func(updateX)

    return updateX, loss


def tau_solve(tol=1e-3, lr=1e-3, device="cuda"):
    # beginTime = time.time()

    initAction = np.array(pd.read_csv('tau.csv', index_col=0))[:, 0]
    initAction = torch.from_numpy(initAction).float()
    initAction = initAction.clone().detach().requires_grad_(True)
    # initAction = torch.tensor(initAction, requires_grad=True, device=device)

    resX, resLoss = tau_descend(initAction, tol=tol, lr=lr)

    # endTime = time.time()

    print("------------------tau--------------------")
    # print("Time: %.3f" % (endTime - beginTime))

    # print("loss: %.8f" % (float(resLoss)))
    return torch.abs(resX.detach())

def get_tau():
    device = torch.device('cpu')
    # create_mat()
    tau = tau_solve(tol=1e-8, lr=1e-3, device=device)
    # print(tau)
    return tau

def opt_bandwidth(b=-1):
    global X, Y
    if b == -1:
        X = np.array(pd.read_csv('0vec_t.csv', index_col=0))
        Y = np.array(pd.read_csv('Y0.csv', index_col=0))
    else:
        X = np.array(pd.read_csv('MSE_outcome/' + str(b) + '_0vec_t.csv', index_col=0))
        Y = np.array(pd.read_csv('MSE_outcome/' + str(b) + '_Y0.csv', index_col=0))

    if b == -1:
        kr = KernelRidge(alpha=1e-3)

        param_dist = {"kernel": [MultiMatern(l1, l2, l3, nu=2.5)
                             for l1 in np.logspace(-3, 3, 15)
                             for l2 in np.logspace(-3, 3, 15)
                             for l3 in np.logspace(-3, 3, 15)]}
        search_cv = RandomizedSearchCV(kr, param_dist, cv=5, n_iter=500, n_jobs=-1)

        '''
        # grid search
        param_dist = {"kernel": [MultiMatern(l1, l2, l3, nu=2.5)
                                 for l1 in np.linspace(0.01, 5, 8)
                                 for l2 in np.linspace(0.01, 5, 8)
                                 for l3 in np.linspace(0.01, 5, 8)]}
        search_cv = GridSearchCV(kr, param_dist, cv=5, n_jobs=-1)
        '''

        search_cv.fit(X, Y)

        best_kernel = search_cv.best_params_['kernel']

        tau = np.zeros([d])
        tau[0] = best_kernel.get_length()['l1']
        tau[1] = best_kernel.get_length()['l2']
        tau[2] = best_kernel.get_length()['l3']

        DF = pd.DataFrame(tau)
        DF.to_csv('tau.csv')
    else:
        tau = get_tau().detach().numpy()

        DF = pd.DataFrame(tau)
        DF.to_csv('MSE_outcome/' + str(b) + '_tau.csv')
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


def generate_wb(d_value: int, s_value: int, b=-1):
    np.random.seed(b + 5)

    global d, s
    d = d_value
    s = s_value

    opt_bandwidth(b)
    if b == -1:
        tau = np.array(pd.read_csv('tau.csv', index_col=0))
    else:
        tau = np.array(pd.read_csv('MSE_outcome/' + str(b) + '_tau.csv', index_col=0))

    w_js = np.zeros([d, s])
    b_js = np.zeros([d, s])
    for i in range(d):
        w_js[i] = sample_p(s, tau[i])
        b_js[i] = np.random.uniform(0, 2 * math.pi, size=s)

    if b == -1:
        DF = pd.DataFrame(w_js)
        DF.to_csv('w_js.csv')
        DF = pd.DataFrame(b_js)
        DF.to_csv('b_js.csv')
    else:
        DF = pd.DataFrame(w_js)
        DF.to_csv('MSE_outcome/' + str(b) + '_w_js.csv')
        DF = pd.DataFrame(b_js)
        DF.to_csv('MSE_outcome/' + str(b) + '_b_js.csv')
    return

if __name__ == '__main__':
    np.random.seed(4)

    # generate_wb(d_value=3, s_value=8)


