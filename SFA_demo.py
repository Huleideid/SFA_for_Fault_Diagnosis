from SFA import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

## 加载数据并标准化
X_train = pd.read_csv("./data/d00.dat", header=None, sep='\s+').values.T
X_test = pd.read_csv("./data/d05_te.dat", header=None, sep='\s+').values
X_train_, X_test_ = normalize(X_train, X_test)
## 训练
MT2, MT2_e, MS2, MS2_e, S2_threshold, S2e_threshold, T2_threshold, T2e_threshold = fit_SFA(X_train_, 0.01)
## 测试
test_T2, test_T2e, test_S2, test_S2e = test_SFA(X_test_, MT2, MT2_e, MS2, MS2_e)
## 监控结果可视化
visualization_SFA(test_T2, test_T2e, test_S2, test_S2e, S2_threshold, S2e_threshold, T2_threshold, T2e_threshold)