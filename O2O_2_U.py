"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Load dataset
data_path = "./data/O2O_2_U/"
pic_path = "./pic/O2O_2_U/"
model_path = "./model/O2O_2_U/"

###################################
# 1 . data process
###################################

ratio = np.array([0.85,0.1,0.05])

x = np.loadtxt(data_path + 'x_10_15.dat')

U = np.loadtxt(data_path + 'U_10_15.dat')
wO2 = np.loadtxt(data_path + 'wO2_10_15.dat')
wO = np.loadtxt(data_path + 'wO_10_15.dat')

N = U.shape[0]
N_train = int(ratio[0] * N)
N_valid = int(ratio[1] * N)
N_test = N - N_train - N_valid

# 需要打乱数据加上这一行,对行数进行打乱
shuffle_ix = np.random.permutation(np.arange(U.shape[0]))
U = U[shuffle_ix]
wO2 = wO2[shuffle_ix]
wO = wO[shuffle_ix]

Ma = U[:,0]
U = U[:,1:-1:10]
wO2 = wO2[:,1:-1:10]
wO = wO[:,1:-1:10]

U_train = U[0:N_train, :]
U_valid = U[N_train:N_train+N_valid, :]
U_test = U[-N_test:, :]
wO2_train = wO2[0:N_train, :]
wO2_valid = wO2[N_train:N_train+N_valid, :]
wO2_test = wO2[-N_test:, :]
wO_train = wO[0:N_train, :]
wO_valid = wO[N_train:N_train+N_valid, :]
wO_test = wO[-N_test:, :]

# 这里的x其实就是传感器的位置，因为输入只需要函数的分布即可，和align有不同
x = x.reshape(-1,1)[0:-1:10]
node = x.shape[0]

# 下面的第二个node是因为我们的U和T取得位置和传感器的位置相同，其实可以是不同，第一个node是我们要预测的位置
input_train0 = np.tile(np.hstack((wO2_train,wO_train)),node).reshape(-1,2*node)
input_train1 = np.tile(x,(N_train,1))
output_train = np.reshape(U_train, (-1,1))

input_valid0 = np.tile(np.hstack((wO2_valid,wO_valid)),node).reshape(-1,2*node)
input_valid1 = np.tile(x,(N_valid,1))
output_valid = np.reshape(U_valid, (-1,1))

input_test0 = np.tile(np.hstack((wO2_test,wO_test)),node).reshape(-1,2*node)
input_test1 = np.tile(x,(N_test,1))
output_test = np.reshape(U_test, (-1,1))


Ma_train = Ma[0:N_train]
Ma_valid = Ma[N_train:N_train+N_valid]
Ma_test = Ma[-N_test:]

# x:Branch and Trunk net input y:output
x_train = (input_train0, input_train1)
y_train = output_train

x_valid = (input_valid0, input_valid1)
y_valid = output_valid

x_test = (input_test0, input_test1)
y_test = output_test

# 构造三元组
data = dde.data.Triple(
    X_train=x_train, y_train=y_train, X_test=x_valid, y_test=y_valid
)

###################################
# 2 . define net
###################################

# Choose a network
n = 10
activation = f"LAAF-{n} relu"
m = x_train[0].shape[1]
dim_x = 1
net = dde.nn.DeepONet(
    [m, 100, 100, 100, 100],
    [dim_x, 100, 100, 100, 100],
    activation,
    "Glorot normal",
    use_bias=True,
    stacked=False,
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.0006, metrics=["mean l2 relative error"])
checker = dde.callbacks.ModelCheckpoint(
    model_path+"model.ckpt", verbose=1, save_better_only=True, period=1000
)
losshistory, train_state = model.train(iterations=120000,
                                       callbacks=[checker])
#                                       model_restore_path=model_path+"model.ckpt-19000.ckpt")

# 保存数据和图片
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

dde.utils.plot_loss_history(losshistory,pic_path+'loss_history.png')
dde.utils.save_loss_history(losshistory,data_path+'loss_history.dat')

###################################
# 3 . result prediction
###################################

# restore model
# model.restore(model_path+"model.ckpt-113000.ckpt")
model.restore(model_path+"model.ckpt-" + str(train_state.best_step) + ".ckpt", verbose=1)

print("Predicting ...")
y_pred = model.predict(x_test)

print("test data L2 relative error:", dde.metrics.l2_relative_error(y_test, y_pred))

y_pred = np.reshape(y_pred,(N_test,node))
y_test = np.reshape(y_test,(N_test,node))
true_U = np.insert(y_test[0:N_test, :],0,Ma_test,axis=1)
pred_U = np.insert(y_pred[0:N_test, :],0,Ma_test,axis=1)
# save
print("Saving prediction data ...")
np.savetxt(data_path+'test_U_pred.dat', pred_U,fmt='%.6f')
np.savetxt(data_path+'test_U_true.dat', true_U,fmt='%.6f')