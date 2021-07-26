import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats


def centerize(*args):
    """
    中心化函数
    对正常数据和测试数据进行中心化，输入数据一般为训练数据矩阵 x_train 和测试数据矩阵 x_test
    （注意：测试数据需要按照正常数据中心化）

    *args: 不定长的输入参数，一般为 2 个
    args[0]: x_train
    args[1]: x_test(如果有的话)

    x_train_center: 中心化后的正常过程数据
    x_test_center: 中心化后的测试过程数据
    """
    
    x_train = args[0]
    x_train_mean = np.mean(x_train, axis=0)
    x_train_center = x_train - x_train_mean

    if len(args) == 2:
        x_test = args[1]
        x_test_center = x_test - x_train_mean
        return (x_train_center, x_test_center)

    if len(args) > 2:
        print("parameter error!")

    return x_train_center

## 归一化数据
def normalize(*args):
    """
    对正常数据和测试数据进行标准化，输入数据一般为训练数据矩阵X_normal和测试数据矩阵X_new
    （注意：测试数据需要按照正常数据的均值和方差标准化）
    """
    X_normal = args[0]
    X_normal_mean = np.mean(X_normal, axis=0)
    X_normal_std = np.std(X_normal, axis=0)
    X_normal_row, X_normal_col = X_normal.shape
    X_normal_center = (X_normal - X_normal_mean) / X_normal_std

    if len(args) == 2:
        X_new = args[1]
        X_new_row, X_new_col = X_new.shape
        X_new_center = (X_new - X_normal_mean) / X_normal_std
        return (X_normal_center, X_new_center)

    return X_normal_center

def fit_SFA(train_X, alpha):
    """
        Establish SFA model
    """

    N, m = train_X.shape
    B = (train_X.T @ train_X) / (N - 1)
    U, lamda, _ = np.linalg.svd(B)
    Lamda = np.diag(lamda)
    Z = train_X @ U@ np.sqrt(np.linalg.inv(Lamda))
    derive_Z = Z[1:, :] - Z[:-1, :]
    cov_derive_Z = (derive_Z.T @ derive_Z) / (N - 2)
    P, omega, _ = np.linalg.svd(cov_derive_Z)
    # resort the result of SVD on cov_derive_Z
    W = P.T @ np.sqrt(np.linalg.inv(Lamda)) @ U.T  # 慢特征变换矩阵
    # Determine the value of M
    Omega = np.diag(omega)
    derive_X = train_X[1:, :] - train_X[:-1, :]
    tmp = (derive_X @ derive_X.T) / (N - 2)
    var_X = np.zeros((m, 1))
    for i in range(m):
        var_X[i] = tmp[i, i]
    max_var = var_X.max()
    M = sum(omega < max_var)
    len_W = W.shape[0]
    Wd = W[len_W - M:, :]
    We = W[:len_W - M, :]
    Omega_d = Omega[len_W - M:, len_W - M:]
    Omega_e = Omega[:len_W - M, :len_W - M]
    MT2 = Wd.T @ Wd
    MT2_e = We.T @ We
    MS2 = Wd.T @ np.linalg.inv(Omega_d) @ Wd
    MS2_e = We.T @ np.linalg.inv(Omega_e) @ We
    Me = m - M
    alpha = 0.01
    level = 1 - alpha
    S2_threshold = scipy.stats.f.ppf(level, M, N - M - 1) * M * (N ** 2 - 2 * N) / (N - 1) / (N - M - 1)
    S2e_threshold = scipy.stats.f.ppf(level, Me, N - Me - 1) * Me * (N ** 2 - 2 * N) / (N - 1) / (N - Me - 1)
    T2_threshold = scipy.stats.chi2.ppf(level, M)
    T2e_threshold =  scipy.stats.chi2.ppf(level, Me)
    return MT2, MT2_e, MS2, MS2_e, S2_threshold, S2e_threshold, T2_threshold, T2e_threshold

def test_SFA(test_X, MT2, MT2_e, MS2, MS2_e):
    test_N, m = test_X.shape
    test_derive_X = test_X[1:, :] - test_X[:-1, :]
    T2 = np.zeros((test_N, ))
    T2e = np.zeros((test_N, ))
    S2e = np.zeros((test_N, )) 
    S2 = np.zeros((test_N, )) 
    for k in range(test_N):
        T2[k] = test_X[k:k+1,:] @ MT2  @ test_X[k:k+1,:].T
        T2e[k] = test_X[k:k+1,:] @ MT2_e @ test_X[k:k+1,:].T
    for k in range(1,test_N):
        S2[k] = test_derive_X[k-1:k, :] @ MS2 @ test_derive_X[k-1:k, :].T
        S2e[k] = test_derive_X[k-1:k, :] @ MS2_e  @ test_derive_X[k-1:k, :].T
    return T2, T2e, S2, S2e


def visualization_SFA(T2, T2e, S2, S2e, T2_threshold, T2e_threshold, S2_threshold, S2e_threshold):
    # Plot Monitoring plots
    test_N =T2.shape[0]
    plt.figure(figsize=(10, 10), dpi=300)
    ax1 = plt.subplot(4,1,1)
    ax1.plot(S2)
    ax1.plot(S2_threshold * np.ones((test_N, 1)), "r--")
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('$S^2$')
    ax2 = plt.subplot(4,1,2)
    ax2.plot(S2e)
    ax2.plot(S2e_threshold * np.ones((test_N, 1)), "r--")
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('$S^2_e$')
    ax3 = plt.subplot(4,1,3)
    ax3.plot(T2)
    ax3.plot(T2_threshold * np.ones((test_N, 1)), "r--")
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('$T^2$')   
    ax4 = plt.subplot(4,1,4)
    ax4.plot(T2e)
    ax4.plot(T2e_threshold * np.ones((test_N, 1)), "r--")
    ax4.set_xlabel('Samples')
    ax4.set_ylabel('$T^2_e$')   
    plt.show()