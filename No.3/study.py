import numpy as np
import scipy.special
import matplotlib.pyplot as plt

Loss_data = []
W = []
V = []
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # ノード数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 行列の計算式
        # w11 w21
        # w12 w22 etc
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # 学習率
        self.lr = learningrate

        # シグモイド関数
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    #学習
    def train(self, inputs_list, targets_list):
        # 入力値と目標値の転置
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 隠れ層の入力値
        hidden_inputs = np.dot(self.wih, inputs)
        # 隠れ層の出力値
        hidden_outputs = self.activation_function(hidden_inputs)

        # 出力層の入力値
        final_inputs = np.dot(self.who, hidden_outputs)
        # 出力層の出力値
        final_outputs = self.activation_function(final_inputs)

        # 目標値と出力層の出力値の誤差
        output_errors = targets - final_outputs
        Loss_data.append(float((output_errors)**2))

        # 重みと誤差の行列計算
        hidden_errors = np.dot(self.who.T, output_errors)

        # P.95
        # 降下勾配法
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_outputs))
        W.append([self.who[0][0], self.who[0][1]])

        # 降下勾配法
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        np.transpose(inputs))
        V.append([self.wih[0][0], self.wih[0][1], self.wih[1][0], self.wih[1][1]])

        pass

    #学習ででた重みを用いる
    def query(self, inputs_list):
        # 入力値
        inputs = np.array(inputs_list, ndmin=2).T

        # 隠れ層の入力値
        hidden_inputs = np.dot(self.wih, inputs)
        # 隠れ層の出力値
        hidden_outputs = self.activation_function(hidden_inputs)

        # 出力層の入力値
        final_inputs = np.dot(self.who, hidden_outputs)
        # 出力層の出力値
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# ノード数
input_nodes = 2
hidden_nodes = 2
output_nodes = 1

# 学習率
learning_rate = 0.1

# ニューラルネットワーク
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)


# エポック数
epochs = 50
E = []

X = [
    [1.211961, -0.17295],
    [1.238369, 0.94545],
    [0.949147, 0.09687],
    [0.999577, 1.171578],
    [1.200233, 0.819478],
    [1.2952, 0.085471],
    [0.049256, 0.268667],
    [0.037686, 1.040169],
    [0.830001, 0.766374],
    [-.025907, -0.15944],
    [1.21624, -0.18398],
    [-.02379, 1.11853],
    [0.940918, 0.080239],
    [0.864679, 1.135235],
    [0.015325, 1.029233],
    [0.8113504, 0.747619],
    [-0.23696, -0.25376],
    [0.796719, 0.902244],
    [1., 1.],
    [1., 0.],
]
y = [0.99, 0.01, 0.99, 0.01, 0.01, 0.99, 0.01, 0.99, 0.01, 0.01, 0.99, 0.99, 0.99, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.99]

X_plot = [X[i][0] for i in range(len(X))]
Y_plot = [X[i][1] for i in range(len(X))]

for e in range(epochs):

    print("epock:{}".format(e+1))
    for i, record in enumerate(X):
        n.train(record, y[i])
        pass
    #pass

# print(W)
# print(V)

# plt.ylim(0, 2)
# plt.plot(range(1000), Loss_data)
# print(Loss_data)

Ans = []
for i, record in enumerate(X):
    y = n.query(record)
    if y > 0.5:
        y = 0.99
    else:
        y = 0.01
    Ans.append(y)
print(Ans)
print(y)
cnt = 0
for i, ans in enumerate(Ans):
    if ans == y[i]:
        cnt += 1

print(cnt / 20)
# x = np.linspace(-2, 2)
# y = (W[-1][0] * (V[-1][0] * x + V[-1][1] * x)) + W[-1][1] * (V[-1][2] * x + V[-1][3] * x)

# plt.scatter(X_plot, Y_plot)
# plt.plot(x, y, color='red')
# plt.show()