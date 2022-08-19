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
    U = np.loadtxt(data_path + 'U_10_15.dat')
    T = np.loadtxt(data_path + 'T_10_15.dat')

    N = x.shape[0]
    N_train = int(ratio[0] * N)
    N_valid = int(ratio[1] * N)
    N_test = N - N_train - N_valid

    data = np.stack((x, U, T), axis=-1)
    # 需要打乱数据加上这一行
    np.random.shuffle(data)

    xx = data[:,0]
    UU = data[:,1]
    TT = data[:,2]
    

    # x:Branch and Trunk net input y:output
    x_train = (data[0:N_train, 1], data[0:N_train, 0])
    y_train = data[0:N_train, 2]

    x_valid = (data[N_train:N_train+N_valid, 1],
            data[N_train:N_train+N_valid, 0])
    y_valid = data[N_train:N_train+N_valid, 2]

    x_test = (data[-N_test:, 1], data[-N_test:, 0])
    y_test = data[-N_test:, 2]
    return x_train,y_train,x_valid,y_valid,x_test,y_test

data_ratio = np.array([0.7,0.2,0.1])
x_train,y_train,x_valid,y_valid,x_test,y_test = generate_data(data_ratio)



data = dde.data.TripleCartesianProd(
    X_train=x_train, y_train=y_train, X_test=x_valid, y_test=y_valid
)

