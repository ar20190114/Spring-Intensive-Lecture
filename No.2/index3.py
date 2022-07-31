from random import random
import numpy as np
import matplotlib.pyplot as plt

# 初期値
N = np.random.rand(10000, 2)
W = np.random.rand(4, 4, 2)
alpha = 0.05

i = 1
while True:

    if i % 20 == 0:
        X = []
        Y = []
        for k in W:
            for l in k:
                X.append(float(l[0]))
                Y.append(float(l[1]))
        plt.scatter(X, Y)
        plt.show()
    
    for (j, n) in enumerate(N):
        min_index = np.argmin(((W - n)**2).sum(axis=2))
        mini = int(min_index / 4)
        minj = int(min_index % 4)

        W[mini][minj] = (1 - alpha) * W[mini][minj] + alpha * n

        for a in range(-1, 2):
            for b in range(-1, 2):
                if a == 0 and b == 0:
                    continue
                elif (mini + a < 0):
                    continue
                elif (mini + a > 3):
                    continue
                elif (minj + b < 0):
                    continue
                elif (minj + b > 3):
                    continue
                else:
                    W[mini+a][minj+b] = (1 - alpha/2) * W[mini][minj] + alpha/2 * n

    if i == 100:
        break

    i += 1