from pandas import DataFrame
from pandas import concat
import pandas as pd
 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
 
raw = pd.read_csv('./stock_data/Stocks/aapl.us.txt')
print(raw.head())
#raw['ob1'] = [x for x in range(10)]
#raw['ob2'] = [x for x in range(50, 60)]
values = raw[['Open', 'Close']].values
print(values[:50])
data = series_to_supervised(values, 1, 2)
print(data)
print(data.shape)
print(data.dtypes)


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time


## assigning predictor and taget variables
x = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
print(x.shape)

data = data.dropna(how='any')

data = data.values
x = data[:, 0:-1]
print(x.shape)
print(type(x))

Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])
print(Y.shape)

Y = data[:, -1]
print(Y.shape)
print(type(Y))


X_test = np.array([[0.42388, 0.42388, 0.42388,0.42134, 0.42516], [173.29000, 174.18000, 174.03000, 175.61000, 174.48000]])#.reshape(1, -1)#, [173.29000, 174.18000, 174.03000, 175.61000, 174.48000]
clf = SVR()
clf.fit(x, Y)
y_pred = clf.predict(X_test)
print(y_pred)





