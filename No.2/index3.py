from random import random
import numpy as np
import matplotlib.pyplot as plt

# 初期値
N = np.random.rand(1500, 3)
W = np.random.rand(4, 4, 3)
alpha = 0.05

i = 1
Colors = ['red', 'blue', 'green', 'pink']
while True:

    if i % 3 == 0:
        X = []
        Y = []
        for k in W:
            for l in k:
                X.append(float(l[0]))
                Y.append(float(l[1]))
        # plt.title("{}".format(i))
        # plt.scatter(X, Y, color = Colors[i//3 - 1])
        # plt.show()
    
    for (j, n) in enumerate(N):
        min_index = np.argmin(((W - n)**2).sum(axis=2))
        mini = int(min_index / 4)
        minj = int(min_index % 4)

        W[mini, minj] = (1 - alpha) * W[mini, minj] + alpha * n

        for a in range(-1, 2):
            for b in range(-1, 2):
                try:
                    if a == 0 and b == 0:
                        continue
                    else:
                        W[mini+a, minj+b] = (1 - alpha/2) * W[mini, minj] + alpha/2 * n
                except:
                    pass

    if i == 12:
        break

    i += 1

print("imgshow")
im = plt.imshow(W,interpolation='none')
plt.show()