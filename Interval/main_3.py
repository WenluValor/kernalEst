import math
import itertools
import numpy as np
import pandas as pd
import data_3 as dt_3
import random
import kernel_3 as knl_3
import time
import torch

def set_global(b: int, s_value: int, n_value: int, d_value, p_value: int):
    random.seed(15)
    np.random.seed(4)
    torch.manual_seed(10)

    global d, n, Y0, Z, s, p, w_js, b_js
    d = d_value # d = 3
    n = n_value
    s = s_value
    p = p_value

    if b == -1:
        Y0 = np.array(pd.read_csv('Y0.csv', index_col=0))[:, 0]
        Y0 = torch.from_numpy(Y0)

        Z = np.array(pd.read_csv('Z.csv', index_col=0))
        Z = torch.from_numpy(Z)

        w_js = np.array(pd.read_csv('w_js.csv', index_col=0))
        w_js = torch.from_numpy(w_js)

        b_js = np.array(pd.read_csv('b_js.csv', index_col=0))
        b_js = torch.from_numpy(b_js)
    else:
        Y0 = np.array(pd.read_csv('outcome/' + str(b) + '_Y0.csv', index_col=0))[:, 0]
        Y0 = torch.from_numpy(Y0)

        Z = np.array(pd.read_csv('outcome/' + str(b) + '_Z.csv', index_col=0))
        Z = torch.from_numpy(Z)

        w_js = np.array(pd.read_csv('outcome/' + str(b) + '_w_js.csv', index_col=0))
        w_js = torch.from_numpy(w_js)

        b_js = np.array(pd.read_csv('outcome/' + str(b) + '_b_js.csv', index_col=0))
        b_js = torch.from_numpy(b_js)
    return

def func(vec_c, data):
    # const_c = vec_c[-1]
    # vec_c = vec_c[0: ((s+1)**d - 1) * (p + 1)]

    lmd = data[0]
    weight = data[1: ]

    sum_main = torch.square(Y0 - torch.matmul(MAT_PSI.double(), vec_c.double()))
    loss_main = torch.mean(sum_main)

    sum_part = torch.zeros([p])
    for j in range(p):
        Y = Z[:, j]

        tmp = torch.square(Y - torch.matmul(PART_PSI_List[j].double(), vec_c.double()))
        sum_part[j] = torch.mean(tmp)
    loss_part = torch.dot(sum_part, weight.float())

    sum_penalty = torch.square(vec_c)
    loss_penalty = torch.sum(sum_penalty) * lmd

    tot_loss = loss_main + loss_part + loss_penalty
    return tot_loss


def descend(initX, data, tol, lr):
    optimizer = torch.optim.Adam([initX], lr=lr)
    updateX = initX * 2 - 1
    loss = func(updateX, data)
    last_loss = 3 * loss.item()
    new_loss = 2 * loss.item()
    count = 0

    while (last_loss - new_loss > tol) | (last_loss < new_loss):
        last_loss = new_loss

        updateX = initX
        loss = func(updateX, data)
        new_loss = loss.item()
        lossABackward = torch.sum(loss)
        optimizer.zero_grad()
        lossABackward.backward()
        optimizer.step()

        print(last_loss)
        print(new_loss)
        print(count)
        print('--------------------==============')
        count += 1

    updateX = initX
    loss = func(updateX, data)

    return updateX, loss


def solve(data, tol=1e-3, lr=1e-3, device="cuda"):
    beginTime = time.time()
    data = data.to(device)
    initAction = torch.rand(((s+1)**d - 1) * (p + 1), requires_grad=True, device=device)

    # initAction = np.array(pd.read_csv('torch_C.csv', index_col=0))[:, 0]
    # initAction = torch.from_numpy(initAction).float()
    # initAction = torch.tensor(initAction, requires_grad=True, device=device)

    resX, resLoss = descend(initAction, data, tol=tol, lr=lr)

    endTime = time.time()

    print("--------------------------------------------")
    print("Time: %.3f" % (endTime - beginTime))

    print("loss: %.8f" % (float(resLoss)))
    return resX

def get_vec_c_torch(lmd, weight: np.array, b=-1):
    global MAT_PSI, PART_PSI_List

    data = torch.cat((lmd, weight), dim=0)
    device = torch.device('cpu')
    if b == -1:
        # create_mat()
        MAT_PSI = np.array(pd.read_csv('mat_PSI.csv', index_col=0))
        MAT_PSI = torch.from_numpy(MAT_PSI)

        PART_PSI_List = [None] * p
        for i in range(p):
            mat_PSI = np.array(pd.read_csv(str(i) + 'part_mat_PSI.csv', index_col=0))
            PART_PSI_List[i] = torch.from_numpy(mat_PSI)
    else:
        # create_mat(b)
        MAT_PSI = np.array(pd.read_csv('outcome/' + str(b) + '_mat_PSI.csv', index_col=0))
        MAT_PSI = torch.from_numpy(MAT_PSI)

        PART_PSI_List = [None] * p
        for i in range(p):
            mat_PSI = np.array(pd.read_csv('outcome/' + str(i) + 'part_mat_PSI.csv', index_col=0))
            PART_PSI_List[i] = torch.from_numpy(mat_PSI)

    vec_c = solve(data, tol=1e-4, lr=1e-2, device=device)
    return vec_c

def create_mat(b=-1):
    mat_PSI = get_pdPSI()
    mat_PSI = mat_PSI.detach().numpy()
    DF = pd.DataFrame(mat_PSI)
    if b == -1:
        DF.to_csv('mat_PSI.csv')
    else:
        DF.to_csv('outcome/' + str(b) + '_mat_PSI.csv')

    for i in range(p):
        mat_PSI = get_partial_pdPSI(i)
        mat_PSI = mat_PSI.detach().numpy()
        DF = pd.DataFrame(mat_PSI)
        if b == -1:
            DF.to_csv(str(i) + 'part_mat_PSI.csv')
        else:
            DF.to_csv('outcome/' + str(i) + 'part_mat_PSI.csv')
    return

def hat_fn_loss(X_test, Y_test, vec_c):
    Y_pred = hat_fn(X_test, vec_c)
    sum_main = torch.square(Y_test - Y_pred)
    loss_main = torch.mean(sum_main)
    return loss_main

def hat_fn(X_test, vec_c):
    if not isinstance(X_test, int):
        mat_PSI = get_pdPSI(vec_t=X_test)
    else:
        if X_test == -1:
            mat_PSI = np.array(pd.read_csv('mat_PSI.csv', index_col=0))
            mat_PSI = torch.from_numpy(mat_PSI)
        else:
            mat_PSI = np.array(pd.read_csv('outcome/' + str(X_test) + '_mat_PSI.csv', index_col=0))
            mat_PSI = torch.from_numpy(mat_PSI)
    return torch.matmul(mat_PSI.double(), vec_c.double())

def hat_partial_fn_loss(X_test, Y_test, vec_c, part_ind):
    Y_pred = hat_partial_fn(X_test, vec_c, part_ind)
    tmp = torch.square(Y_test - Y_pred)
    sum_part = torch.mean(tmp)
    return sum_part

def hat_partial_fn(X_test, vec_c, part_ind):
    if not isinstance(X_test, int):
        mat_PSI = get_partial_pdPSI(part_ind, vec_t=X_test)
    else:
        if X_test == -1:
            mat_PSI = np.array(pd.read_csv(str(part_ind) + 'part_mat_PSI.csv', index_col=0))
            mat_PSI = torch.from_numpy(mat_PSI)
        else:
            mat_PSI = np.array(pd.read_csv('outcome/' + str(part_ind) + 'part_mat_PSI.csv', index_col=0))
            mat_PSI = torch.from_numpy(mat_PSI)
    return torch.matmul(mat_PSI.double(), vec_c.double())

def get_pdPSI(vec_t=1):
    if isinstance(vec_t, int):
        vec_t = np.array(pd.read_csv(str(0) + 'vec_t.csv', index_col=0))
        # vec_t *= 100
        vec_t = torch.from_numpy(vec_t)

    PSI = pdPSI(vec_t[0]).reshape(1, -1)
    for i in range(1, vec_t.shape[0]):
        tmp = pdPSI(vec_t[i]).reshape(1, -1)
        PSI = torch.cat((PSI, tmp), 0)
    return PSI

def get_partial_pdPSI(part_ind, vec_t=1):
    if isinstance(vec_t, int):
        vec_t = np.array(pd.read_csv(str(part_ind + 1) + 'vec_t.csv', index_col=0))
        # vec_t *= 100
        vec_t = torch.from_numpy(vec_t)

    PSI = partial_pdPSI(vec_t[0], part_ind).reshape(1, -1)
    for i in range(1, vec_t.shape[0]):
        tmp = partial_pdPSI(vec_t[i], part_ind).reshape(1, -1)
        PSI = torch.cat((PSI, tmp), 0)
    return PSI

def partial_pdPSI(t, part_ind: int):
    PSI = first_diff_psi_d(t, part_ind)
    for i in range(p):
        PSI = torch.cat((PSI, second_diff_psi_d(t, i, part_ind)), 0)
    return PSI

def pdPSI(t):
    # t = (t1, t2, t4) is not existing data
    PSI = psi_d(t)
    for i in range(p):
        PSI = torch.cat((PSI, first_diff_psi_d(t, i)), 0)
    return PSI

def second_diff_psi_d(t, j1: int, j2: int):
    # j1, j2 = 0, 1, ... p-1
    mat_cos = tilde_psi(t, 0)
    mat_first = tilde_psi(t, 1)
    mat_zero = torch.zeros([d, s])

    ind_set = []
    for i in range(t.shape[0]):
        j = i + 1
        B = list(itertools.combinations(np.arange(t.shape[0]), j))
        ind_set = ind_set + B

    ans = torch.tensor([])
    if j1 == j2:
        ans = first_diff_psi_d(t, part_ind=j1, is_two_order=True)
        return ans

    for i in range(len(ind_set)):
        if (j1 in ind_set[i]) & (j2 in ind_set[i]):
            ind_i = int(ind_set[i][0])
            if (ind_set[i][0] == j1) | (ind_set[i][0] == j2):
                tmp = mat_first[ind_i]
            else:
                tmp = mat_cos[ind_i]

            pos1 = ind_set[i].index(j1)
            pos2 = ind_set[i].index(j2)
            for l in range(1, len(ind_set[i])):
                ind_i = int(ind_set[i][l])
                if (l == pos1) | (l == pos2):
                    tmp = torch.kron(tmp, mat_first[ind_i])
                else:
                    tmp = torch.kron(tmp, mat_cos[ind_i])
        else:
            ind_i = int(ind_set[i][0])
            tmp = mat_zero[ind_i]
            for l in range(1, len(ind_set[i])):
                ind_i = int(ind_set[i][l])
                tmp = torch.kron(tmp, mat_zero[ind_i])

        ans = torch.cat((ans, tmp), 0)
    return ans

def first_diff_psi_d(t, part_ind: int, is_two_order=False):
    # part_ind = 0, 1, ... p-1
    mat_cos = tilde_psi(t, 0)
    mat_first = tilde_psi(t, 1)
    if is_two_order:
        mat_first = tilde_psi(t, 2)
    mat_zero = torch.zeros([d, s])

    ind_set = []
    for i in range(t.shape[0]):
        j = i + 1
        B = list(itertools.combinations(np.arange(t.shape[0]), j))
        ind_set = ind_set + B

    ans = torch.tensor([])
    for i in range(len(ind_set)):
        if part_ind in ind_set[i]:
            ind_i = int(ind_set[i][0])
            if ind_set[i][0] == part_ind:
                tmp = mat_first[part_ind]
            else:
                tmp = mat_cos[ind_i]

            pos = ind_set[i].index(part_ind)
            for l in range(1, len(ind_set[i])):
                ind_i = int(ind_set[i][l])
                if l == pos:
                    tmp = torch.kron(tmp, mat_first[part_ind])
                else:
                    tmp = torch.kron(tmp, mat_cos[ind_i])
        else:
            ind_i = int(ind_set[i][0])
            tmp = mat_zero[ind_i]
            for l in range(1, len(ind_set[i])):
                ind_i = int(ind_set[i][l])
                tmp = torch.kron(tmp, mat_zero[ind_i])

        ans = torch.cat((ans, tmp), 0)
    return ans

def psi_d(t):
    # t = torch, size = 3
    mat_cos = tilde_psi(t, 0)

    ind_set = []
    for i in range(t.shape[0]):
        j = i + 1
        B = list(itertools.combinations(np.arange(t.shape[0]), j))
        ind_set = ind_set + B

    ans = torch.tensor([])
    for i in range(len(ind_set)):
        ind_i = int(ind_set[i][0])
        tmp = mat_cos[ind_i]
        for l in range(1, len(ind_set[i])):
            ind_i = int(ind_set[i][l])
            tmp = torch.kron(tmp, mat_cos[ind_i])

        ans = torch.cat((ans, tmp), 0)
    return ans


def tilde_psi(t, order):
    # t of shape (d, )
    # order = 0, 1, 2
    # each row is s-dim random feature
    new_t = torch.kron(t.reshape(-1, 1), torch.ones(s))
    ans = new_t * w_js + b_js
    if order == 0:
        return torch.cos(ans) * math.sqrt(2 / s)
    elif order == 1:
        return -torch.sin(ans) * w_js * math.sqrt(2 / s)
    elif order == 2:
        return -torch.cos(ans) * w_js * w_js * math.sqrt(2 / s)
    return


def boots_compute(N: int, s: int, d: int, p: int, b=-1):
    set_global(b=b, s_value=s, n_value=N, d_value=d, p_value=p)
    lmd = get_gcv_lmd()
    weight = torch.ones([p])
    for i in range(p):
        weight[i] = torch.var(Y0) / torch.var(Z[:, i])

    C = get_vec_c_torch(lmd=lmd, weight=weight, b=b)
    C = C.detach().numpy()
    DF = pd.DataFrame(C)
    DF.to_csv('outcome/' + str(b) + '_torch_C.csv')
    return


def boots_hat_fn(t, N: int, s: int, d: int, p: int, b=-1):
    set_global(b=b, s_value=s, n_value=N, d_value=d, p_value=p)

    if b == -1:
        vec_c = np.array(pd.read_csv('torch_C.csv', index_col=0))[:, 0]
        vec_c = torch.from_numpy(vec_c).float()
    else:
        vec_c = np.array(pd.read_csv('outcome/' + str(b) + '_torch_C.csv', index_col=0))[:, 0]
        vec_c = torch.from_numpy(vec_c).float()

    return hat_fn(X_test=t, vec_c=vec_c.reshape(-1, 1))


def boots_partial_fn(t, part_ind, N: int, s: int, d: int, p: int, b=-1):
    set_global(b, s_value=s, n_value=N, d_value=d, p_value=p)

    if b == -1:
        vec_c = np.array(pd.read_csv('torch_C.csv', index_col=0))[:, 0]
        vec_c = torch.from_numpy(vec_c).float()
    else:
        vec_c = np.array(pd.read_csv('outcome/' + str(b) + '_torch_C.csv', index_col=0))[:, 0]
        vec_c = torch.from_numpy(vec_c).float()

    return hat_partial_fn(X_test=t, part_ind=part_ind, vec_c=vec_c.reshape(-1, 1))

def gcv_func(lmd):
    vec_x = np.array(pd.read_csv(str(0) + 'vec_t.csv', index_col=0))
    vec_x = torch.from_numpy(vec_x)
    for i in range(p):
        tmp = np.array(pd.read_csv(str(i + 1) + 'vec_t.csv', index_col=0))
        tmp = torch.from_numpy(tmp)
        vec_x = torch.cat((vec_x, tmp), dim=0)
    num = vec_x.shape[0]

    vec_y = Y0
    for i in range(p):
        vec_y = torch.cat((vec_y, Z[:, i]), dim=0)

    mat_lmd = torch.matmul(torch.transpose(vec_x, 0, 1), vec_x) + num * lmd * torch.eye(d)
    mat_lmd = torch.linalg.inv(mat_lmd)
    mat_A = torch.matmul(vec_x, torch.matmul(mat_lmd, torch.transpose(vec_x, 0, 1)))
    mat_A = torch.eye(num) - mat_A

    hat_y = torch.matmul(mat_A.double(), vec_y.double())

    numerator = torch.square(torch.norm(hat_y, p=2))
    denominator = torch.square(torch.trace(mat_A)) / num

    return numerator / denominator

def gcv_descend(initX, tol, lr):
    optimizer = torch.optim.Adam([initX], lr=lr)
    updateX = initX * 2 - 1
    loss = gcv_func(updateX)
    last_loss = 3 * loss.item()
    new_loss = 2 * loss.item()
    count = 0

    while (last_loss - new_loss > tol) | (last_loss < new_loss):
        last_loss = new_loss

        updateX = initX
        loss = gcv_func(updateX)
        new_loss = loss.item()
        lossABackward = torch.sum(loss)
        optimizer.zero_grad()
        lossABackward.backward()
        optimizer.step()

        print(last_loss)
        print(new_loss)
        print(count)
        print('-----------GCV=========')
        count += 1

    updateX = initX
    loss = gcv_func(updateX)

    return updateX, loss


def gcv_solve(tol=1e-3, lr=1e-3, device="cuda"):
    beginTime = time.time()
    torch.manual_seed(10)
    initAction = torch.rand(1, requires_grad=True, device=device)

    resX, resLoss = gcv_descend(initAction, tol=tol, lr=lr)

    endTime = time.time()

    print("------------------GCV--------------------")
    print("Time: %.3f" % (endTime - beginTime))

    print("loss: %.8f" % (float(resLoss)))
    return resX

def get_gcv_lmd():
    device = torch.device('cpu')
    # create_mat()
    lmd = gcv_solve(tol=1e-3, lr=1e-3, device=device)
    return lmd


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    random.seed(15)
    np.random.seed(4)
    set_global(s_value=8,
               n_value=1000, d_value=3, p_value=3)
    lmd = get_gcv_lmd()
    weight = torch.ones([p])
    for i in range(p):
        weight[i] = torch.var(Y0) / torch.var(Z[:, i])

    C = get_vec_c_torch(lmd, weight)
    C = C.detach().numpy()
    DF = pd.DataFrame(C)
    DF.to_csv('torch_C.csv')

    C = np.array(pd.read_csv('torch_C.csv', index_col=0))[:, 0]
    C = torch.from_numpy(C).float()

    size = 1000
    A = np.zeros([size])
    time = 10
    B = np.zeros(time)

    np.random.seed(7)
    # vec_t = np.array(pd.read_csv('0vec_t.csv', index_col=0))
    for j in range(time):
        t = torch.rand((size, d))
        # t *= 100
        Y_test = torch.zeros([size])

        for i in range(size):
            Y_test[i] = torch.from_numpy(dt_3.true_f(t[i]))

        B[j] = hat_fn_loss(X_test=t, Y_test=Y_test, vec_c=C)
        print(B[j])
        print('p1n1000-----')

    print(np.mean(B))
    print(np.std(B))
    print('-=-=-=-=-==-=-=-=')

    exit(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
