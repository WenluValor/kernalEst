import itertools
import numpy as np
import pandas as pd

def set_global(B: int):
    vec_eps = np.array(pd.read_csv('vec_epsilon.csv', index_col=0))

    global n, p, eps
    n = vec_eps.shape[0]
    p = vec_eps.shape[1] - 1
    eps = np.zeros([n, p + 1, B])

    for b in range(-1, B - 1):
        if b == -1:
            vec_eps = np.array(pd.read_csv('vec_epsilon.csv', index_col=0))
        else:
            vec_eps = np.array(pd.read_csv('MSE_outcome/' + str(b) + '_vec_epsilon.csv', index_col=0))

        eps[:, :, b] = vec_eps
    return

def get_all_cov():
    ans = np.zeros([n - 1])
    # ind_list = []
    ind_list = list(itertools.combinations(np.arange(p + 1), 2))
    for i in range(p + 1):
        ind_list.append((i, i))

    for l in range(1, n):
        print(l)
        print('----')
        ans[l - 1] = get_cov(ind_list=ind_list, length=l)
    return ans


def get_cov(ind_list: list, length: int):
    ans = np.zeros([len(ind_list)])
    i = 0
    for ind in ind_list:
        ans[i] = get_single_cov(ind=ind, length=length)
        i += 1
    print(ans)
    return np.mean(ans)


def get_single_cov(ind, length: int):
    ans1 = np.zeros([n - length])
    ans2 = np.zeros([n - length])
    for i in range(n - length):
        a = eps[i, int(ind[0]), :]
        b = eps[i + length, int(ind[1]), :]
        ans1[i] = np.cov(a, b)[0][1]

        a = eps[i, int(ind[1]), :]
        b = eps[i + length, int(ind[0]), :]
        ans2[i] = np.cov(a, b)[0][1]

    ans = (np.mean(ans1) + np.mean(ans2)) / 2
    return ans


def plot_err():
    y = get_all_cov()
    x = np.arange(1, y.shape[0] + 1)
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.savefig('Eg2p1n10r0.png')
    plt.show()



if __name__ == '__main__':
    B = 200
    set_global(B=B)
    plot_err()


    exit(0)