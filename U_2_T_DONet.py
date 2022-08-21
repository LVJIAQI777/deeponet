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

Ma = U[:,0]
U = U[:,1:]
T = T[:,1:]

x = x.reshape(-1,1)

Ma_train = Ma[0:N_train]
Ma_valid = Ma[N_train:N_train+N_valid]
Ma_test = Ma[-N_test:]

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
checker = dde.callbacks.ModelCheckpoint(
    model_path+"model.ckpt", verbose=1, save_better_only=True, period=1000
)
losshistory, train_state = model.train(iterations=120000, callbacks=[checker])

print("Best step : {}".format(train_state.best_step))
print("Best step valid data MSE : {}".format(train_state.best_metrics[0]))

# 保存数据和图片
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

dde.utils.plot_loss_history(losshistory,pic_path+'loss_history.png')
dde.utils.save_loss_history(losshistory,data_path+'loss_history.dat')

# 对DeepOnet中triple进行储存
train_U = np.insert(train_state.X_train[0],0,Ma_train,axis=1)
train_T = np.insert(train_state.y_train,0,Ma_train,axis=1)
valid_U = np.insert(train_state.X_test[0],0,Ma_valid,axis=1)
valid_T = np.insert(train_state.y_test,0,Ma_valid,axis=1)
valid_T_pred = np.insert(train_state.best_y,0,Ma_valid,axis=1)
test_U = np.insert(x_test[0],0,Ma_test,axis=1)
test_T = np.insert(y_test,0,Ma_test,axis=1)

fname_train_U = data_path+'train_U.dat'
print("Saving training data to {} ...".format(fname_train_U))
np.savetxt(fname_train_U, train_U,fmt='%.6f')

fname_train_T = data_path+'train_T.dat'
print("Saving training data to {} ...".format(fname_train_T))
np.savetxt(fname_train_T, train_T,fmt='%.6f')

fname_valid_U = data_path+'valid_U.dat'
print("Saving valid data to {} ...".format(fname_valid_U))
np.savetxt(fname_valid_U, valid_U,fmt='%.6f')

fname_valid_T = data_path+'valid_T.dat'
print("Saving valid data to {} ...".format(fname_valid_T))
np.savetxt(fname_valid_T, valid_T,fmt='%.6f')

fname_valid_T_pred = data_path+'valid_T_pred.dat'
print("Saving valid data to {} ...".format(fname_valid_T_pred))
np.savetxt(fname_valid_T_pred, valid_T_pred,fmt='%.6f')

fname_test_U = data_path+'test_U.dat'
print("Saving test data to {} ...".format(fname_test_U))
np.savetxt(fname_test_U, test_U,fmt='%.6f')

fname_test_T = data_path+'test_T.dat'
print("Saving test data to {} ...".format(fname_test_T))
np.savetxt(fname_test_T, test_T,fmt='%.6f')