# https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

# import data
# 
# data was already cleaned and prepared, 
# meaning missing stock and index prices 
# were LOCF'ed (last observation carried 
# forward), so that the file did not 
# contain any missing values
data = pd.read_csv('./sp500/data_stocks.csv')

# drop data variable
data = data.drop(['DATE'], 1)

# dimensions of dataset
n = data.shape[0]
p = data.shape[1]

plt.plot(data['SP500'])
plt.show()


# make data a numpy array
data = data.values



# training and test data 
train_start = 0
train_end = int(np.floor(0.8*n))

test_start = train_end
test_end = n

data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]


# scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

# build X and Y
X_train = data_train[:, 1:]
Y_train = data_train[:, 0]
X_test = data_test[:, 1:]
Y_test = data_test[:, 0]


# placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# model architecture parameters
n_stocks = 500
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

# Layer 1: variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))


