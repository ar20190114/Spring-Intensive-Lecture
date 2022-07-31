import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, fun, der, xi=0.3, tau=0.9, tol=1e-6, ite_max=2000):
        self.fun = fun         # 目的関数
        self.der = der         # 関数の勾配
        self.xi  = xi          # Armijo条件の定数
        self.tau = tau         # 方向微係数の学習率
        self.tol = tol         # 勾配ベクトルのL2ノルムがこの値より小さくなると計算を停止
        self.path = None       # 解の点列
        self.ite_max = ite_max # 最大反復回数
        
    def minimize(self, x):
        path = [x]
        
        for i in range(self.ite_max):
            grad = self.der(x)
            
            if np.linalg.norm(grad, ord=2)<self.tol:
                break
            else:
                beta = 1
                
                while self.fun(x - beta*grad) > (self.fun(x) - self.xi*beta*np.dot(grad, grad)):
                    # Armijo条件を満たすまでループする
                    beta = self.tau*beta
                    
                x = x - beta * grad
                path.append(x)
        
        self.opt_x = x                # 最適解
        self.opt_result = self.fun(x) # 関数の最小値
        self.path = np.array(path)    # 探索解の推移

def f(x):
    return 2*x[0]**2 + x[1]**2 + x[0]*x[1]

def f_der(x):
    return np.array([4*x[0] + x[1], x[0] + 2*x[1]])

x1 = np.linspace(-2, 2, 21)
x2 = np.linspace(-2, 2, 21)
x1_mesh, x2_mesh = np.meshgrid(x1, x2)
z = f(np.array((x1_mesh, x2_mesh)))

fig, ax = plt.subplots(figsize=(6, 6))
ax.contour(x1, x2, z, levels=np.logspace(-0.3, 1.2, 10))
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_aspect('equal')
plt.show()


x1 = np.linspace(-2, 2, 21)
x2 = np.linspace(-2, 2, 21)
x1_mesh, x2_mesh = np.meshgrid(x1, x2)

grad = f_der(np.array((x1_mesh, x2_mesh)))
U = grad[0] # x1方向の勾配
V = grad[1] # x2方向の勾配

fig, ax = plt.subplots(figsize=(6, 6))
ax.quiver(x1, x2, U, V, color='blue')
ax.set_aspect('equal')
plt.show()

gd = GradientDescent(f, f_der)
init = np.array([1.5, 1.5])
gd.minimize(init)

path = gd.path

x1 = np.linspace(-2, 2, 21)
x2 = np.linspace(-2, 2, 21)
x1_mesh, x2_mesh = np.meshgrid(x1, x2)
z = f(np.array((x1_mesh, x2_mesh)))

fig, ax = plt.subplots(figsize=(6, 6))
ax.contour(x1, x2, z, levels=np.logspace(-0.3, 1.2, 10))
ax.plot(path[:,0], path[:,1], marker="o")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_aspect('equal')
plt.show()
