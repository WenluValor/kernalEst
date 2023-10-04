import numpy as np
import math
import pandas as pd
from scipy import linalg

def generate_data(p_value: int, n_value: int, d_value: int, up: float, down: float):
    global x_data, z_data, p, n, d
    x_data = np.array([-1.71479842809193, -1.02165124753198, -0.579818495252942, -0.198450938723838, 0.0953101798043249,
                  0.350656871613169, 0.598836501088704, 0.824175442966349, 1.03673688495002, 1.23547147138531,
                  1.43031124653667, 1.62136648329937, 1.80500469597808, 1.98513086220859, 2.16332302566054,
                  2.33795223683134, 2.50959926237837, 2.67965072658051, 2.84954976337591])
    z_data = np.array([0.0538181818181818, 0.0878181818181818, 0.121090909090909, 0.135636363636364, 0.118363636363636,
                  0.0900000000000000, 0.0649090909090909, 0.0470909090909091, 0.0338181818181818, 0.0243636363636364,
                  0.0174545454545455, 0.0118181818181818, 0.00890909090909091, 0.00600000000000000, 0.00454545454545455,
                  0.00272727272727273, 0.00236363636363636, 0.00181818181818182, 0.00127272727272727])
    p = p_value
    n = n_value
    d = d_value

    for i in range(p + 1):
        vec_t = np.random.uniform(down, up, size=(n, d))
        DF = pd.DataFrame(vec_t)
        DF.to_csv(str(i) + 'vec_t.csv')

    Y0 = np.zeros([n])
    vec_t = np.array(pd.read_csv('0vec_t.csv', index_col=0))
    for i in range(n):
        ind = np.random.choice(np.array(x_data.shape[0]), 10)
        ind = list(ind)
        Y0[i] = Y_0(t=vec_t[i], ind=ind)
    DF = pd.DataFrame(Y0)
    DF.to_csv('Y0.csv')

    Z = np.zeros([n, p])
    for j in range(p):
        vec_t = np.array(pd.read_csv(str(j + 1) + 'vec_t.csv', index_col=0))
        for i in range(n):
            ind = np.random.choice(np.array(x_data.shape[0]), 10)
            ind = list(ind)
            Z[i, j] = partial_Y(t=vec_t[i], j=j + 1, ind=ind)
    DF = pd.DataFrame(Z)
    DF.to_csv('Z.csv')

def A(t: np.array):
    t1 = t[0]
    t2 = t[1]
    t3 = t[2]
    ans = np.array([[-t2 - t3, t1, 0, 0],
                    [t2, -t1 - t2, t1, 0],
                    [0, t2, -t1 - t2, t1],
                    [0, 0, t2, -t1]])
    return ans

def eta(x, t: np.array):
    mat_A = math.exp(x) * A(t)
    e1 = np.array([1, 0, 0, 0])
    e4 = np.array([0, 0, 0, 1])
    ans = np.dot(np.dot(e1, linalg.expm(mat_A)), e4)
    return ans

def Y_0(t: np.array, ind: list):
    x_dt = x_data[ind]
    z_dt = z_data[ind]
    sum = 0
    for i in range(x_dt.shape[0]):
        sum += (z_dt[i] - eta(x_dt[i], t))**2
    sum /= x_dt.shape[0]
    return sum

def partial_A(t: np.array, j: int):
    # j =  1, 2, 3
    t1 = t[0]
    t2 = t[1]
    t3 = t[2]
    if j == 1:
        ans = np.array([[0, 1, 0, 0],
                        [0, -1, 1, 0],
                        [0, 0, -1, 1],
                        [0, 0, 0, -1]])
    elif j == 2:
        ans = np.array([[-1, 0, 0, 0],
                        [1, -1, 0, 0],
                        [0, 1, -1, 0],
                        [0, 0, 1, 0]])
    else:
        ans = np.array([[-1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
    return ans

def partial_eta(x, t: np.array, j: int):
    mat_A = math.exp(x) * partial_A(t, j)
    e1 = np.array([1, 0, 0, 0])
    e4 = np.array([0, 0, 0, 1])
    ans = np.dot(np.dot(e1, linalg.expm(mat_A)), e4)
    return ans

def partial_Y(t: np.array, j: int, ind: list):
    # j = 1, 2, 3
    sum = 0
    x_dt = x_data[ind]
    z_dt = z_data[ind]
    for i in range(x_dt.shape[0]):
        sum += (z_dt[i] - eta(x_dt[i], t)) * partial_eta(x_dt[i], t, j)
    sum *= -2 / x_dt.shape[0]
    return sum

def true_f(t: np.array):
    x_data = np.array([-1.71479842809193, -1.02165124753198, -0.579818495252942, -0.198450938723838, 0.0953101798043249,
                       0.350656871613169, 0.598836501088704, 0.824175442966349, 1.03673688495002, 1.23547147138531,
                       1.43031124653667, 1.62136648329937, 1.80500469597808, 1.98513086220859, 2.16332302566054,
                       2.33795223683134, 2.50959926237837, 2.67965072658051, 2.84954976337591])

    z_data = np.array([0.0538181818181818, 0.0878181818181818, 0.121090909090909, 0.135636363636364, 0.118363636363636,
                       0.0900000000000000, 0.0649090909090909, 0.0470909090909091, 0.0338181818181818,
                       0.0243636363636364,
                       0.0174545454545455, 0.0118181818181818, 0.00890909090909091, 0.00600000000000000,
                       0.00454545454545455,
                       0.00272727272727273, 0.00236363636363636, 0.00181818181818182, 0.00127272727272727])
    sum = 0
    for i in range(x_data.shape[0]):
        sum += (z_data[i] - eta(x_data[i], t))**2
    sum /= x_data.shape[0]
    return sum