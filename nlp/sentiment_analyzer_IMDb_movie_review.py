# credit article: https://towardsdatascience.com/machine-learning-text-processing-1d5a2d638958
# http://ai.stanford.edu/~amaas/data/sentiment/

# convert the dataset from files to a python DataFrame
import os
import pandas as pd

"""
folder = 'aclImdb'

labels = {'pos': 1, 'neg': 0}

df = pd.DataFrame()

for f in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(folder, f, l)
        for file_ in os.listdir(path):
            with open(os.path.join(path,file_), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],ignore_index=True)


df.columns = ['review', 'sentiment']

df.to_csv('movie_data.csv', index=False, encoding='utf-8')"""

df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head())

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

reviews = df.review.str.cat(sep=' ')

# function to split text into word
tokens = word_tokenize(reviews)
stop_words = set(stopwords.words('english'))
print(stop_words)
tokens = [w for w in tokens if not w.lower() in stop_words and len(w) > 2]

vocabulary = set(tokens)
print(len(vocabulary))

frequency_dist = nltk.FreqDist(tokens)
print(sorted(frequency_dist, key=frequency_dist.__getitem__, reverse=True)[0:50])

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud().generate_from_frequencies(frequency_dist)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()

X_train = df.loc[:24999, 'review'].values
Y_train = df.loc[:24999, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
Y_test = df.loc[25000:, 'sentiment'].values

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

print(train_vectors.shape, test_vectors.shape)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(train_vectors, Y_train)

from sklearn.metrics import accuracy_score
predicted = clf.predict(test_vectors)
print(accuracy_score(Y_test, predicted))

