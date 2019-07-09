# Autoregression example
from statsmodels.tsa.ar_model import AR
from random import random

import matplotlib.pyplot as plt

# contrived dataset
data = [x + random() for x in range(1,100)]

# fit the model
model = AR(data)

model_fit = model.fit()

# make predication
yhat = model_fit.predict(len(data), len(data))
print(data)
print(yhat)


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(range(len(data)), data, 'o',label="data")
ax.plot(len(data)+ 1, yhat, 'P',label="Predicted")
plt.show()


