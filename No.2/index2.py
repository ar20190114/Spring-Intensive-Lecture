from turtle import distance
import numpy as np
import re
import copy


def logData(min_index, w, d):
    WD1, WD2, WD3, WD4, WD5 = [], [], [], [], []

    if min_index == 0:
        WD1.append([copy.deepcopy(w), copy.deepcopy(d)])

    elif min_index == 1:
        WD2.append([copy.deepcopy(w), copy.deepcopy(d)])

    elif min_index == 2:
        WD3.append([copy.deepcopy(w), copy.deepcopy(d)])

    elif min_index==3:
        WD4.append([copy.deepcopy(w), copy.deepcopy(d)])

    elif min_index == 4:
        WD5.append([copy.deepcopy(w), copy.deepcopy(d)])

    return WD1, WD2, WD3, WD4, WD5


# データ取得
x = []
with open('20220625.txt', 'r') as f:
    for i,line in enumerate(f.readlines()):
        if i >= 1:
            a = re.split('[\t]',line)
            b = [float(a[1]),float(a[2])]
            x.append(b)

x = np.array(x)

# 環境変数
finish_point = 0.01
alpha = 0.05

# 初期値
w = np.array([1. ,1.])
w_log = []
WD1, _, _, _, _ = logData(0, w[0], 0)

while True:
    D = np.zeros(len(w))

    while True:
        deltaW = np.zeros(w.shape)

        for i, x_i in enumerate(x):

            min_index = np.argmin((abs(w - x)).sum(axis=1))

            print(x[i])
            print(w)
            d = np.linalg.norm(x[i] - w[min_index], ord=2)
            deltaW[min_index] = alpha * (x[i] - w[min_index])

            w[min_index] = w[min_index] + deltaW[min_index]
            D[min_index] = 0.9 * D[min_index] + 0.1 * d

        if np.all(np.abs(deltaW) < 0.01) == True:
            break
        w_log.append(len(WD1))
    
    if np.max(D) < 0.01:
        break

    max_index = np.argmax(D)
    w = np.block([w], w[max_index])

print(w)