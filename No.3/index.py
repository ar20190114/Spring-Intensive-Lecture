import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# data
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
]
y = [0.99, 0.01, 0.99, 0.01, 0.01, 0.99, 0.01, 0.99, 0.01, 0.01, 0.99, 0.99, 0.99, 0.01, 0.99, 0.01, 0.01, 0.01]

# tensor変換
x = torch.tensor(X, dtype=torch.float32)
t = torch.tensor(y, dtype=torch.float32) 

# dataset
dataset = TensorDataset(x,t)
# data loader
train = dataset
batch_size = 2 # mini batch size
train_loader = DataLoader(train, batch_size, shuffle=True)

# nn
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        mid = 2 # mid layer
        # input:2 mid:2
        self.fc1 = nn.Linear(2,mid)
        # mid:2 output:1
        self.fc2 = nn.Linear(mid,1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x
net = Net()

print("Initial weights")
print(net.fc1.weight.data)
print(net.fc1.bias.data)
print(net.fc2.weight.data)
print(net.fc2.bias.data)

# Loss
criterion = nn.MSELoss(reduction="sum")

# Stochastic gradient descent
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Train
print("+++TRAIN+++")
max_epoch=10000
Loss_data = []

for epoch in range(max_epoch+1) :
    for batch in train_loader:
        x,t = batch
        #clear grad
        optimizer.zero_grad() 
        #forward
        y=net(x)
        #loss function
        loss = criterion(y,t)
        Loss_data.append(float(loss))
        #BP
        loss.backward()
        #update
        optimizer.step()
    if epoch % 50 == 0:
        print("epoc:", epoch, ' loss:', loss.item())

print("Final weights")
print(net.fc1.weight.data)
print(net.fc1.bias.data)
print(net.fc2.weight.data)
print(net.fc2.bias.data)

X_plot = [X[i][0] for i in range(len(X))]
Y_plot = [X[i][1] for i in range(len(X))]

print(len(Loss_data))

plt.ylim(0, 2)
plt.plot(range(90009), Loss_data)

# plt.scatter(X_plot, Y_plot)
plt.show()