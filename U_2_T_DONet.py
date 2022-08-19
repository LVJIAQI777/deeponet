"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data_path = "./data/U_2_T/"
pic_path = "./pic/U_2_T/"
model_path = "./model/U_2_T/"

def generate_data(ratio):
    x = np.loadtxt(data_path + 'x_10_15.dat')
    U = np.loadtxt(data_path + 'U_10_15.dat')[:,1:]
    T = np.loadtxt(data_path + 'T_10_15.dat')[:,1:]

    N = U.shape[0]
    N_train = int(ratio[0] * N)
    N_valid = int(ratio[1] * N)
    N_test = N - N_train - N_valid
    
    x = x.reshape(-1,1)

    # x:Branch and Trunk net input y:output
    x_train = (U[0:N_train, :], x)
    y_train = T[0:N_train, :]

    x_valid = (U[N_train:N_train+N_valid, :], x)
    y_valid = T[N_train:N_train+N_valid, :]

    x_test = (U[-N_test:, :], x)
    y_test = T[-N_test:, :]
    return x_train,y_train,x_valid,y_valid,x_test,y_test

data_ratio = np.array([0.7,0.2,0.1])
x_train,y_train,x_valid,y_valid,x_test,y_test = generate_data(data_ratio)

# 构造三元组
data = dde.data.TripleCartesianProd(
    X_train=x_train, y_train=y_train, X_test=x_valid, y_test=y_valid
)

# Choose a network
m = x_train[1].shape[0]
dim_x = 1
net = dde.nn.DeepONetCartesianProd(
    [m, 100, 100, 100],
    [dim_x, 100, 100, 100],
    "relu",
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
checker = dde.callbacks.ModelCheckpoint(
    model_path+"model.ckpt", verbose=1, save_better_only=True, period=1000
)
losshistory, train_state = model.train(iterations=50000, callbacks=[checker])

# 保存数据和图片
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

dde.utils.plot_loss_history(losshistory,pic_path+'loss_history.png')
dde.utils.save_loss_history(losshistory,data_path+'loss_history.dat')