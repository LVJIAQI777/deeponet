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