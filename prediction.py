"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data_path = "./data/U_2_T/"
pic_path = "./pic/U_2_T/"
model_path = "./model/U_2_T/"

ratio = np.array([0.7,0.2,0.1])

x = np.loadtxt(data_path + 'x_10_15.dat')

U = np.loadtxt(data_path + 'U_10_15.dat')
T = np.loadtxt(data_path + 'T_10_15.dat')

N = U.shape[0]
N_train = int(ratio[0] * N)
N_valid = int(ratio[1] * N)
N_test = N - N_train - N_valid

# 需要打乱数据加上这一行,对行数进行打乱
shuffle_ix = np.random.permutation(np.arange(U.shape[0]))
U = U[shuffle_ix]
T = T[shuffle_ix]

U = U[:,1:]
T = T[:,1:]

x = x.reshape(-1,1)

# x:Branch and Trunk net input y:output
x_train = (U[0:N_train, :], x)
y_train = T[0:N_train, :]

x_valid = (U[N_train:N_train+N_valid, :], x)
y_valid = T[N_train:N_train+N_valid, :]

x_test = (U[-N_test:, :], x)
y_test = T[-N_test:, :]

# 构造三元组
data = dde.data.TripleCartesianProd(
    X_train=x_train, y_train=y_train, X_test=x_valid, y_test=y_valid
)

# Choose a network
m = x_train[1].shape[0]
dim_x = 1
net = dde.nn.DeepONetCartesianProd(
    [m, 100, 100, 100, 100],
    [dim_x, 100, 100, 100, 100],
    "relu",
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.0006, metrics=["mean l2 relative error"])

'''
predict
'''
# restore model
model.restore(model_path+"model.ckpt-113000.ckpt")

U = np.loadtxt(data_path + 'test_U.dat')
Ma = U[:,0]
U = np.delete(U,0,axis=1)
x = np.loadtxt(data_path + 'x_10_15.dat')
x = x.reshape(-1,1)

inputdata = (U,x)
outputdata = model.predict(inputdata)
pred_T = np.insert(outputdata,0,Ma,axis=1)

# save
fname_output = data_path+'test_T_pred.dat'
print("Saving test data to {} ...".format(fname_output))
np.savetxt(fname_output, pred_T,fmt='%.6f')

