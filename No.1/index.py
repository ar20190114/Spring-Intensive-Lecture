# パーセプトロン分類

import numpy as np
import matplotlib.pyplot as plt
import re

# パーセプトロン
def Perceptron(x, w, beta):
    ans = np.dot(w, x) - beta
    if ans <= 0:
        return 0
    else:
        return 1

# データの読み込み
X, y = [], []
x_range0, y_range0 = [], []
x_range1, y_range1 = [], []
with open('20220611.txt', 'r') as f:
    for i,line in enumerate(f.readlines()):
        if i >= 1:
            a = re.split('[\t]',line)
            b = [float(a[1]),float(a[2])]
            c = int(a[3])
            X.append(b)
            y.append(c)

            if c == 0:
                x_range0.append(float(a[1]))
                y_range0.append(float(a[2]))
            else:
                x_range1.append(float(a[1]))
                y_range1.append(float(a[2]))

# 初期値
X = np.array(X)
w = np.array([np.random.rand(), np.random.rand()])
beta = np.random.rand()
epock = 1
learning_rate = 0.05

# 学習
while True:
    acc = len(X)

    for i, x in enumerate(X):
        y_pre = Perceptron(x, w, beta)
        delta = y_pre - y[i]

        if delta == 0:
            acc -= 1

        # w, betaの更新
        w -= learning_rate * delta * x
        beta += learning_rate * delta

    if acc == 0:
        break
    epock += 1

print(epock)
print(w)
print(beta)
plt.scatter(x_range0, y_range0, color='red')
plt.scatter(x_range1, y_range1, color='blue')
x_range = np.linspace(-1, 2)
y = -(w[0] * x_range - beta) / w[1]
plt.plot(x_range, y, color='black')
plt.show()