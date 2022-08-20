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
    x = np.insert(x,0,[-1])
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
    [m, 100, 100, 100, 100],
    [dim_x, 100, 100, 100, 100],
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
losshistory, train_state = model.train(iterations=100000, callbacks=[checker])

print("Best step valid data MSE:{}".format(train_state.best_metrics[0]))

# 保存数据和图片
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

dde.utils.plot_loss_history(losshistory,pic_path+'loss_history.png')
dde.utils.save_loss_history(losshistory,data_path+'loss_history.dat')

# 对DeepOnet中triple进行储存
fname_train_U = data_path+'train_U.dat'
print("Saving training data to {} ...".format(fname_train_U))
np.savetxt(fname_train_U, train_state.X_train[0],fmt='%.6f')

fname_train_T = data_path+'train_T.dat'
print("Saving training data to {} ...".format(fname_train_T))
np.savetxt(fname_train_T, train_state.y_train,fmt='%.6f')

fname_valid_U = data_path+'valid_U.dat'
print("Saving valid data to {} ...".format(fname_valid_U))
np.savetxt(fname_valid_U, train_state.X_test[0],fmt='%.6f')

fname_valid_T = data_path+'valid_T.dat'
print("Saving valid data to {} ...".format(fname_valid_T))
np.savetxt(fname_valid_T, train_state.y_test,fmt='%.6f')

fname_valid_T_pred = data_path+'valid_T_pred.dat'
print("Saving valid data to {} ...".format(fname_valid_T_pred))
np.savetxt(fname_valid_T_pred, train_state.best_y,fmt='%.6f')

fname_test_U = data_path+'test_U.dat'
print("Saving test data to {} ...".format(fname_test_U))
np.savetxt(fname_test_U, x_test[0],fmt='%.6f')

fname_test_T = data_path+'test_T.dat'
print("Saving test data to {} ...".format(fname_test_T))
np.savetxt(fname_test_T, y_test,fmt='%.6f')