"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data_path = "./data/"
pic_path = "./pic/"
model_path = "./model/"

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

    # x:Branch and Trunk net input y:output
    xx = data[:, 0]
    uu = data[:, 1]

    xx = xx.reshape(-1,1)
    uu = uu.reshape(-1,1)

    # sensor
    ss = np.ones([1,1])

    x_train = (xx[0:N_train, :], ss)
    y_train = uu[0:N_train, :]

    x_valid = (xx[N_train:N_train+N_valid, :], ss)
    y_valid = uu[N_train:N_train+N_valid, :]

    x_test = (xx[-N_test:, :], ss)
    y_test= uu[-N_test:, :]

    return x_train, y_train, x_valid, y_valid, x_test, y_test

data_ratio = np.array([0.2,0.2,0.1])
x_train,y_train,x_valid,y_valid,x_test,y_test = generate_data(data_ratio)

# 构造三元组
data = dde.data.TripleCartesianProd(
    X_train=x_train, y_train=y_train, X_test=x_valid, y_test=y_valid
)

# Choose a network
m = 1
dim_x = 1
net = dde.nn.DeepONetCartesianProd(
    [m, 40, 40],
    [dim_x, 40, 40],
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
losshistory, train_state = model.train(iterations=20000, callbacks=[checker])

# 保存数据和图片
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

dde.utils.plot_loss_history(losshistory,pic_path+'loss_history.png')
dde.utils.save_loss_history(losshistory,data_path+'loss_history.dat')

# 对DeepOnet中triple进行储存
fname_train = data_path+'train.dat'
fname_test = data_path+'test.dat'
print("Saving training data to {} ...".format(fname_train))
train = np.hstack((train_state.X_train[0], train_state.y_train))
np.savetxt(fname_train, train, header="x, y")

print("Saving test data to {} ...".format(fname_test))
test = np.hstack((train_state.X_test[0], train_state.y_test, train_state.best_y))
np.savetxt(fname_test, test, header="x, y_true, y_pred")

# 对DeepOnet中best_state进行绘图并保存
y_train = train_state.y_train
y_test = train_state.y_test
best_y = train_state.best_y
y_dim = best_y.shape[1]

idx = np.argsort(train_state.X_test[0][:, 0])
X = train_state.X_test[0][idx, 0]
plt.figure()
for i in range(y_dim):
    if y_train is not None:
        plt.plot(train_state.X_train[0][:, 0], y_train[:, i], "ok", label="Train")
    if y_test is not None:
        plt.plot(X, y_test[idx, i], "-k", label="True")
    plt.plot(X, best_y[idx, i], "--r", label="Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# dde.utils.save_best_state(train_state,data_path+"train.dat",data_path+"test.dat")
# dde.utils.plot_best_state(train_state)
plt.savefig(pic_path+"best_state.jpg")