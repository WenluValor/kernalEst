import math
import itertools
import numpy as np
import pandas as pd
import data_2 as dt_2
import random
import kernel_2 as knl_2
import time
import torch

def set_global(c_value: np.array, vec_t_up: float, vec_t_low: float, s_value: int,
               rho_value: float, n_value: int, std_value: float, mean_value: float, p_value: int):
    global d, c, rho, n, vec_epsilon, Y0, Z, s, p
    d = c_value.shape[0] - 1 # d = 3
    c = c_value
    n = n_value
    s = s_value
    p = p_value


    dt_2.generate_data(c_value=c_value, d_value=d, vec_t_up=vec_t_up, vec_t_low=vec_t_low, rho_value=rho_value,
                       n_value=n_value, std_value=std_value, mean_value=mean_value, p_value=p)
    # vec_t = (t_1, t_2, 1, t_4)
    vec_epsilon = np.array(pd.read_csv('vec_epsilon.csv', index_col=0))
    Y0 = np.array(pd.read_csv('Y0.csv', index_col=0))[:, 0]
    Z = np.array(pd.read_csv('Z.csv', index_col=0))

    knl_2.generate_wb(d=d, s=s, low=vec_t_low-vec_t_up, high=vec_t_up-vec_t_low)

    global w_js, b_js
    w_js = np.array(pd.read_csv('w_js.csv', index_col=0))
    b_js = np.array(pd.read_csv('b_js.csv', index_col=0))

    return

def func(vec_c, data: np.array):
# for torch_func
    lmd = data[0].item()
    weight = data[1: ].detach().numpy()
    # vec_c = vec_c.detach().numpy()
    # vec_c = vec_c.reshape(-1)

    sum = 0
    vec_t = np.array(pd.read_csv('0vec_t.csv', index_col=0))
    vec_t = np.delete(vec_t, 2, axis=1)

    mean_Y0 = np.mean(Y0)
    # vec_t = (t1, t2, t4)

    for i in range(n):
        t = vec_t[i]
        sum += (Y0[i] - mean_Y0 + hat_fn(t, vec_c))**2

    sum /= n

    sub_sum = 0
    for j in range(p):
        subsub_sum = 0
        Y = Z[:, j]
        mean_Y = np.mean(Y)
        vec_t = np.array(pd.read_csv(str(j + 1) + 'vec_t.csv', index_col=0))
        vec_t = np.delete(vec_t, 2, axis=1)

        for i in range(n):
            t = vec_t[i]
            subsub_sum += (Y[i] - mean_Y - partial_hat_fn(t, vec_c, j))**2

        subsub_sum /= n
        sub_sum += weight[j] * subsub_sum

    '''
    print(sub_sum)
    print('---')
    print(sum)
    print('===')
    '''

    sum += sub_sum

    sub_sum = 0
    for i in range(vec_c.shape[0]):
        sub_sum += vec_c[i]**2

    sum += lmd * sub_sum
    return sum


def descend(initX, data, tol, lr):
    optimizer = torch.optim.Adam([initX], lr=lr)
    # optimizer = torch.optim.SGD([initX], lr=lr, momentum=0.9)
    updateX = initX * 2 - 1
    loss = func(updateX, data)
    last_loss = 3 * loss.item()
    new_loss = 2 * loss.item()
    count = 0

    while (last_loss - new_loss > tol) | (last_loss < new_loss):
        last_loss = new_loss

        updateX = initX * 2 - 1
        loss = func(updateX, data)
        new_loss = loss.item()
        lossABackward = torch.sum(loss)
        optimizer.zero_grad()
        lossABackward.backward()
        # loss.backward()
        optimizer.step()

        print(last_loss)
        print(new_loss)
        print(count)
        count += 1

    '''
    for i in range(0, step):
        updateX = initX * 2 - 1
        loss = func(updateX, data)
        lossABackward = torch.sum(loss)
        optimizer.zero_grad()
        lossABackward.backward()
        # loss.backward()
        optimizer.step()
        print(loss.item())
    '''

    updateX = initX * 2 - 1
    loss = func(updateX, data)
    #minId = torch.argmin(loss)
    #print(minId)

    return updateX, loss


def solve(data, tol=1e-3, lr=1e-3, device="cuda"):
    beginTime = time.time()
    data = data.to(device)
    torch.manual_seed(6)
    initAction = torch.rand(((s+1)**d - 1) * (p + 1), requires_grad=True, device=device)
    resX, resLoss = descend(initAction, data, tol=tol, lr=lr)

    endTime = time.time()

    print("--------------------------------------------")
    print("Time: %.3f" % (endTime - beginTime))
    # print("a=%.5f, b=%.5f, c=%.5f" % (float(data[0]), float(data[1]), float(data[2])))

    # print("x:    %.5f" % (float(resX)))
    print("loss: %.8f" % (float(resLoss)))
    return resX

def get_vec_c_torch(lmd: float, weight: np.array):
    data = np.hstack((lmd, weight))
    data = torch.tensor(data)
    device = torch.device('cpu')
    vec_c = solve(data, tol=3e-2, lr=5e-2, device=device)
    return vec_c

def partial_hat_fn(t: np.array, vec_c: np.array, part_ind: int):

    PSI = first_partial_psi_d(t, part_ind)

    for i in range(p):
        PSI = np.hstack((PSI, second_partial_psi_d(t, i, part_ind)))
    PSI = torch.from_numpy(PSI).float()
    ans = torch.dot(PSI, vec_c)
    return ans

def hat_fn(t: np.array, vec_c):
    # t = (t1, t2, t4) is not existing data
    PSI = psi_d(t)
    for i in range(p):
        PSI = np.hstack((PSI, first_partial_psi_d(t, i)))
    PSI = torch.from_numpy(PSI).float()
    ans = torch.dot(PSI, vec_c)
    return ans

def first_partial_psi_d(t: np.array, j: int):
    # j = 0, 1 (p-1)
    # t = (t1, t2, t4)
    index = []
    ind_set = np.zeros([d])
    # ind_set = [[1, 0, 0], [2, 0, 0], ...]
    A = np.arange(d)
    psi = psi_d(t)
    # A = [0, 1, 2]
    for i in range(d):
        size = i + 1
        B = list(itertools.combinations(A, size))
        index = index + B
    for item in index:
        tmp = get_index(list(item))

        ind_set = np.vstack((ind_set, tmp))

    ind_set = np.delete(ind_set, 0, axis=0)
    for i in range(ind_set.shape[0]):
        if (ind_set[i, j] != 0):
            v = int(ind_set[i, j] - 1)
            psi[i] /= math.sqrt(2 / s) * math.cos(t[j] * w_js[j, v] + b_js[j, v])
            psi[i] *= -math.sqrt(2 / s) * w_js[j, v] * math.sin(t[j] * w_js[j, v] + b_js[j, v])
        else:
            psi[i] = 0
    return psi

def second_partial_psi_d(t: np.array, j1: int, j2: int):
    # j1 = 0, 1; j2 = 0, 1
    index = []
    ind_set = np.zeros([d])
    # ind_set = [[1, 0, 0], [2, 0, 0], ...]
    A = np.arange(d)
    psi = psi_d(t)
    # A = [0, 1, 2]
    for i in range(d):
        size = i + 1
        B = list(itertools.combinations(A, size))
        index = index + B

    for item in index:
        tmp = get_index(list(item))
        ind_set = np.vstack((ind_set, tmp))

    ind_set = np.delete(ind_set, 0, axis=0)

    for i in range(ind_set.shape[0]):
        if (ind_set[i, j1] != 0) & (ind_set[i, j2] != 0):
            if (j1 != j2):
                v = int(ind_set[i, j1] - 1)
                part1 = -math.sqrt(2 / s) * w_js[j1, v] * math.sin(t[j1] * w_js[j1, v] + b_js[j1, v])
                psi[i] /= math.sqrt(2 / s) * math.cos(t[j1] * w_js[j1, v] + b_js[j1, v])

                v = int(ind_set[i, j2] - 1)
                part2 = -math.sqrt(2 / s) * w_js[j2, v] * math.sin(t[j2] * w_js[j2, v] + b_js[j2, v])
                psi[i] /= math.sqrt(2 / s) * math.cos(t[j2] * w_js[j2, v] + b_js[j2, v])

                psi[i] *= part1 * part2
            else:
                v = int(ind_set[i, j1] - 1)
                psi[i] *= -w_js[j1, v]**2
        else: psi[i] = 0
    return psi

def get_index(ind: list):
    # ind = [0, 1, 2] / [1, 2]
    size = len(ind)
    ans = np.zeros([s**size, d])

    for i in range(size):
        row = 0
        col = ind[i]
        thold = s**(size - i - 1)
        while (row < s**size):
            for j in range(s):
                count = 0
                while (count < thold):
                    ans[row, col] = j + 1
                    row += 1
                    count += 1
    return ans

def psi_d(t: np.array):
    # t = (t1, t2, t4)
    comb = []
    for i in range(t.shape[0]):
        j = i + 1
        A = list(itertools.combinations(t, j))
        comb = comb + A

    ans = np.array([])
    for sub_t in comb:
        sub_t = np.array(sub_t)
        ans = np.hstack((ans, psi_order(sub_t)))
    return ans

def psi_order(t: np.array):
    # order = 1, 2, 3
    # t = (t1, t2, t4) / (t1, t2)
    order = t.shape[0]
    tilde = [None] * order
    for i in range(order):
        tilde[i] = tilde_psi(t[i], i)

    ans = tilde[0]
    for i in range(order - 1):
        ans = np.kron(ans, tilde[i + 1])
    return ans

def tilde_psi(t: float, j: int):
    # j = 0, 1, 2, j = 2 means take valus in t_4
    ans = np.zeros([s])
    for i in range(s):
        ans[i] = math.sqrt(2 / s) * math.cos(t * w_js[j, i] + b_js[j, i])
    return ans

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    random.seed(15)
    np.random.seed(4)
    set_global(c_value=np.array([1, 0.8, 0.7, 0.6]), vec_t_low=0.5, vec_t_up=1.5, s_value=5,
               rho_value=0.9, n_value=1000, std_value=0.35, mean_value=0, p_value=0)
    lmd = 0.5
    weight = np.ones([p])


    C = get_vec_c_torch(lmd, weight)
    C = C.detach().numpy()
    DF = pd.DataFrame(C)
    DF.to_csv('torch_C.csv')


    C = np.array(pd.read_csv('torch_C.csv', index_col=0))[:, 0]
    C = torch.from_numpy(C).float()
    size = 10000
    A = np.zeros([size])
    '''
    np.random.seed(8)
    for i in range(size):
        t = np.random.uniform(0.5, 1.5, size=3)
        A[i] = hat_fn(t, vec_c=C).item()
    print(np.mean(A))
    print('-=-=-=-=-==-=-=-=')
    '''



    count = 0
    while (np.mean(A) > 0.003) | (np.mean(A) < 0.0023):
        np.random.seed(count)
        for i in range(size):
            t = np.random.uniform(0.5, 1.5, size=3)
            A[i] = hat_fn(t, vec_c=C).item()
        count += 1
        print(np.mean(A))
        print(count - 1)
        print('-=-=-=-=-==-=-=-=')





    exit(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
