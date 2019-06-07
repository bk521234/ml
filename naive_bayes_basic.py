# import library of Gaussion Maive Bates medil
import os

from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time


# assigning predictor and taget variables
x = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])

Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

# create a Gaussiaan Classifier
model = GaussianNB()

# train the model using the training sets
model.fit(x, Y)

# predict output
predicted = model.predict([[1,2], [3,4]])

print(predicted)

