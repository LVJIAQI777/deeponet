"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data_path = "./data/x_2_U/"
pic_path = "./pic/x_2_U/"
model_path = "./model/x_2_U/"


def generate_data(ratio):
    x = np.loadtxt(data_path + 'U.txt')[:, 0]
    U = np.loadtxt(data_path + 'U.txt')[:, 1]
    U = U/U[0]

    N = x.shape[0]
    N_train = int(ratio[0] * N)
    N_valid = int(ratio[1] * N)
    N_test = N - N_train - N_valid

    data = np.stack((x, U), axis=-1)
    # 需要打乱数据加上这一行
    np.random.shuffle(data)

    x_train = data[0:N_train, 0]
    y_train = data[0:N_train, 1]

    x_valid = data[N_train:N_train+N_valid, 0]
    y_valid = data[N_train:N_train+N_valid, 1]

    x_test = data[-N_test:, 0]
    y_test = data[-N_test:, 1]

    # 将1D数组变成2D
    x_train = x_train.reshape(-1,1)
    y_train = y_train.reshape(-1,1)
    x_valid = x_valid.reshape(-1,1)
    y_valid = y_valid.reshape(-1,1)
    x_test = x_test.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


data_ratio = np.array([0.2, 0.1, 0.1])
x_train, y_train, x_valid, y_valid, x_test, y_test = generate_data(data_ratio)

# 设置训练数据
data = dde.data.DataSet(
    X_train=x_train,
    y_train=y_train,
    X_test=x_valid,
    y_test=y_valid,
    standardize=True,
)

# 设置Net
layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)

# 设置模型
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
checker = dde.callbacks.ModelCheckpoint(
    model_path+"model.ckpt", verbose=1, save_better_only=True, period=1000
)
losshistory, train_state = model.train(iterations=20000, callbacks=[checker])

# 保存数据和图片
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

dde.utils.plot_loss_history(losshistory,pic_path+'loss_history.png')
dde.utils.save_loss_history(losshistory,data_path+'loss_history.dat')
dde.utils.save_best_state(train_state,data_path+"train.dat",data_path+"test.dat")
dde.utils.plot_best_state(train_state)
plt.savefig(pic_path+"best_state.jpg")