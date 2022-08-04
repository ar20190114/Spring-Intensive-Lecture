import numpy as np
import re
import matplotlib.pyplot as plt
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D

#記録
Wlog1,Wlog2,Wlog3,Wlog4,Wlog5 = [],[],[],[],[]
D1,D2,D3,D4,D5 = [],[],[],[],[]
def log(c,w,d):
    if c==0:
        Wlog1.append(deepcopy(w))
        D1.append(deepcopy(d))
    elif c==1:
        Wlog2.append(deepcopy(w))
        D2.append(deepcopy(d))
    elif c==2:
        Wlog3.append(deepcopy(w))
        D3.append(deepcopy(d))
    elif c==3:
        Wlog4.append(deepcopy(w))
        D4.append(deepcopy(d))
    elif c==4:
        Wlog5.append(deepcopy(w))
        D5.append(deepcopy(d))

#データの読み込み
x = []
with open('20220625.txt', 'r') as f:
    for i,line in enumerate(f.readlines()):
        if i==0:
            pass
        else:
            a = re.split('[\t]',line)
            b = [float(a[1]),float(a[2])]
            x.append(b)
            np.array(x)

#競合学習
w = np.array([[1.,1.]])
beta = 0.1
copy_eps = np.array([0.01,0.01])
w_eps, D_eps, alpha = 0.1, 0.5, 0.05
wlen = []
log(0,w[0],0.)

while (1):
    D = np.zeros(len(w))

    while (1):
        w_vari = np.zeros(w.shape)

        for j in range(len(x)):
            min_search = []
            for i in range(len(w)):
                distance = np.linalg.norm(x[j]-w[i], ord=2)
                min_search.append(distance)
            C = np.argmin(min_search)
            d = np.linalg.norm(x[j] - w[C], ord=2)
            w_vari[C] = alpha * (x[j] - w[C])
            w[C] = w[C] + w_vari[C]
            D[C] = (1 - beta) * D[C] + beta * d
            log(C,w[C],D[C])

        if np.all(np.abs(w_vari) < w_eps) == True:
            break
        wlen.append(len(Wlog1))

    if np.max(D) < D_eps:
        print('Done')
        break
    k = np.argmax(D)
    w = np.block([[w], [w[k]+copy_eps]])

print(w)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('time')

w1, w2 = [], []
for i in range(len(Wlog1)):
    w1.append(Wlog1[i][0])
    w2.append(Wlog1[i][1])
ax.plot(w1,w2,np.arange(len(Wlog1)),color='b',label='cell1')

print(len(Wlog1))

w1, w2 = [], []
for i in range(len(Wlog2)):
    w1.append(Wlog2[i][0])
    w2.append(Wlog2[i][1])
ax.plot(w1,w2,np.arange(len(Wlog2)),color='g',label='cell2')

w1, w2 = [], []
for i in range(len(Wlog3)):
    w1.append(Wlog3[i][0])
    w2.append(Wlog3[i][1])
ax.plot(w1,w2,np.arange(len(Wlog3)),color='r',label='cell3')

w1, w2 = [], []
for i in range(len(Wlog4)):
    w1.append(Wlog4[i][0])
    w2.append(Wlog4[i][1])
ax.plot(w1,w2,np.arange(len(Wlog4)),color='y',label='cell4')

w1, w2 = [], []
for i in range(len(Wlog5)):
    w1.append(Wlog5[i][0])
    w2.append(Wlog5[i][1])
ax.plot(w1,w2,np.arange(len(Wlog5)),color='c',label='cell5')

ax.view_init(elev=20, azim=40)
x1, x2, w1, w2 = [], [], [], []
for i in range(len(x)):
    x1.append(x[i][0])
    x2.append(x[i][1])

for j in range(len(w)):
    w1.append(w[j][0])
    w2.append(w[j][1])

ax.scatter(x1,x2,color='c')
ax.scatter(w1,w2,color='r')
ax.legend(loc='lower right',fontsize=15)
plt.show()