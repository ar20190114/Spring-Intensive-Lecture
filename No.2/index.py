import numpy as np
import re

#データの読み込み
X = []
with open('20220625.txt', 'r') as f:
    for i,line in enumerate(f.readlines()):
        if i==0:
            pass
        else:
            a = re.split('[\t]',line)
            b = [float(a[1]),float(a[2])]
            X.append(b)
            np.array(X)

print(X)

# 初期値
W = np.array([1., 1.])
alpha = 0.05

