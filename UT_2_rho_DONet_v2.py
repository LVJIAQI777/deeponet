"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
# import tensorflow.compat.v1 as tf

# Load dataset
data_path = "./data/UT_2_rho_nl/"
pic_path = "./pic/UT_2_rho_nl/"
model_path = "./model/UT_2_rho_nl/"

ratio = np.array([0.8,0.15,0.05])

x = np.loadtxt(data_path + 'x_10_15.dat')

U = np.loadtxt(data_path + 'U_10_15.dat')
T = np.loadtxt(data_path + 'T_10_15.dat')
wO2 = np.loadtxt(data_path + 'wO2_10_15_no_log.dat')
wO = np.loadtxt(data_path + 'wO_10_15_no_log.dat')

N = U.shape[0]
N_train = int(ratio[0] * N)
N_valid = int(ratio[1] * N)
N_test = N - N_train - N_valid

# 需要打乱数据加上这一行,对行数进行打乱
shuffle_ix = np.random.permutation(np.arange(U.shape[0]))
U = U[shuffle_ix]
T = T[shuffle_ix]
wO2 = wO2[shuffle_ix]
wO = wO[shuffle_ix]

Ma = U[:,0]
U = U[:,1:-1:5]
T = T[:,1:-1:5]
wO2 = wO2[:,1:-1:5]
wO = wO[:,1:-1:5]

U_train = U[0:N_train, :]
U_valid = U[N_train:N_train+N_valid, :]
U_test = U[-N_test:, :]
T_train = T[0:N_train, :]
T_valid = T[N_train:N_train+N_valid, :]
T_test = T[-N_test:, :]
wO2_train = wO2[0:N_train, :]
wO2_valid = wO2[N_train:N_train+N_valid, :]
wO2_test = wO2[-N_test:, :]
wO_train = wO[0:N_train, :]
wO_valid = wO[N_train:N_train+N_valid, :]
wO_test = wO[-N_test:, :]

input_train = np.concatenate((U_train,T_train), axis=0)
output_train = np.concatenate((wO2_train,wO_train), axis=0)
input_valid = np.concatenate((U_valid,T_valid), axis=0)
output_valid = np.concatenate((wO2_valid,wO_valid), axis=0)
input_test = np.concatenate((U_test,T_test), axis=0)
output_test = np.concatenate((wO2_test,wO_test), axis=0)

x = x.reshape(-1,1)[0:-1:5]

Ma_train = Ma[0:N_train]
Ma_valid = Ma[N_train:N_train+N_valid]
Ma_test = Ma[-N_test:]

# x:Branch and Trunk net input y:output
x_train = (input_train, x)
y_train = output_train

x_valid = (input_valid, x)
y_valid = output_valid

x_test = (input_test, x)
y_test = output_test

# 构造三元组
data = dde.data.TripleCartesianProd(
    X_train=x_train, y_train=y_train, X_test=x_valid, y_test=y_valid
)

# Choose a network
n = 10
activation = f"LAAF-{n} relu"
m = x_train[1].shape[0]
dim_x = 1
net = dde.nn.DeepONetCartesianProd(
    [m, 100, 100, 100, 100],
    [dim_x, 100, 100, 100, 100],
    activation,
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.00001, metrics=["mean l2 relative error"])
checker = dde.callbacks.ModelCheckpoint(
    model_path+"model.ckpt", verbose=1, save_better_only=True, period=1000
)
losshistory, train_state = model.train(iterations=100000, callbacks=[checker], model_restore_path=model_path+"model.ckpt-42000.ckpt")#, model_restore_path=model_path+"model.ckpt-197000.ckpt"

# model.compile("L-BFGS-B", metrics=["mean l2 relative error"])
# losshistory, train_state = model.train(iterations=10000, callbacks=[checker])
# print("Best step : {}".format(train_state.best_step))
# print("Best step valid data mean l2 relative error : {}\n".format(train_state.best_metrics[0]))

# 保存数据和图片
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

dde.utils.plot_loss_history(losshistory,pic_path+'loss_history.png')
dde.utils.save_loss_history(losshistory,data_path+'loss_history.dat')

# 对DeepOnet中triple进行储存
train_U = np.insert(train_state.X_train[0][0:N_train, :],0,Ma_train,axis=1)
train_T = np.insert(train_state.X_train[0][N_train:2*N_train, :],0,Ma_train,axis=1)
train_wO2 = np.insert(train_state.y_train[0:N_train, :],0,Ma_train,axis=1)
train_wO = np.insert(train_state.y_train[N_train:2*N_train, :],0,Ma_train,axis=1)
valid_U = np.insert(train_state.X_test[0][0:N_valid, :],0,Ma_valid,axis=1)
valid_T = np.insert(train_state.X_test[0][N_valid:2*N_valid, :],0,Ma_valid,axis=1)
valid_wO2 = np.insert(train_state.y_test[0:N_valid, :],0,Ma_valid,axis=1)
valid_wO = np.insert(train_state.y_test[N_valid:2*N_valid, :],0,Ma_valid,axis=1)
test_U = np.insert(input_test[0:N_test, :],0,Ma_test,axis=1)
test_T = np.insert(input_test[N_test:2*N_test, :],0,Ma_test,axis=1)
test_wO2 = np.insert(output_test[0:N_test, :],0,Ma_test,axis=1)
test_wO = np.insert(output_test[N_test:2*N_test, :],0,Ma_test,axis=1)

np.savetxt(data_path+'train_U.dat', train_U,fmt='%.6f')
np.savetxt(data_path+'train_T.dat', train_T,fmt='%.6f')
np.savetxt(data_path+'train_wO2.dat', train_wO2,fmt='%.6f')
np.savetxt(data_path+'train_wO.dat', train_wO,fmt='%.6f')
np.savetxt(data_path+'valid_U.dat', valid_U,fmt='%.6f')
np.savetxt(data_path+'valid_T.dat', valid_T,fmt='%.6f')
np.savetxt(data_path+'valid_wO2.dat', valid_wO2,fmt='%.6f')
np.savetxt(data_path+'valid_wO.dat', valid_wO,fmt='%.6f')
np.savetxt(data_path+'test_U.dat', test_U,fmt='%.6f')
np.savetxt(data_path+'test_T.dat', test_T,fmt='%.6f')
np.savetxt(data_path+'test_wO2.dat', test_wO2,fmt='%.6f')
np.savetxt(data_path+'test_wO.dat', test_wO,fmt='%.6f')

# restore model
# model.restore(model_path+"model.ckpt-113000.ckpt")
model.restore(model_path+"model.ckpt-" + str(train_state.best_step) + ".ckpt", verbose=1)

print("Predicting ...")
outputdata = model.predict(x_test)
pred_wO2 = np.insert(outputdata[0:N_test, :],0,Ma_test,axis=1)
pred_wO = np.insert(outputdata[N_test:2*N_test, :],0,Ma_test,axis=1)

print("test data L2 relative error:", dde.metrics.l2_relative_error(y_test, outputdata))

# save
print("Saving prediction data ...")
np.savetxt(data_path+'test_wO2_pred.dat', pred_wO2,fmt='%.6f')
np.savetxt(data_path+'test_wO_pred.dat', pred_wO,fmt='%.6f')