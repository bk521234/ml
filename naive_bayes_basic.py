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


df = pd.read_csv('training_compd_headers.csv')

x_data = df['Raw_Headers'].tolist()
y_data = df['Classification'].tolist()


vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(x_data)  
print(counts.shape)
print(type(counts))
print(counts.toarray())


print(len(x_data))
print(len(y_data))

model2 = GaussianNB()
model2.fit(counts.toarray(), y_data)

# read in files from last year
COMPD_2018_ROOT = r"S:\2018\CompData (COMPD-XXX18)"
COMPD_2017_ROOT = r"S:\2017\CompData (COMPD-XXX17)"

file_count = 0
for root, dirs, files in os.walk(COMPD_2017_ROOT):
    for file in files:
        if file.endswith('.csv'):
            try:
                print(file)
                df = pd.read_csv(os.path.join(root, file), nrows=5, header=None, dtype=str)
                print(os.path.join(root, file))
                print(df.head())

                row_list_of_lists = df.values.tolist()
                cleaned_list = [';ZZZZ'.join(l) for l in row_list_of_lists]

                for i in cleaned_list:
                    vector = vectorizer.transform([i])
                    predicted2 = model2.predict(vector.toarray())
                    print(predicted2)

                    if predicted2 == ['1']:
                        with open('detecting_headers_2017results.txt', 'a') as f:
                            f.write('Detected Headers:\n{}\n{}\n{}\n'.format(os.path.join(root, file), predicted2, i))
                        print(i)
                        print(predicted2)
                    else:
                        with open('detecting_headers_2017results_skipped.txt', 'a') as f:
                            f.write('Skipped Headers:\n{}\n{}\n{}\n'.format(os.path.join(root, file), predicted2, i))


                """
                if file_count == 0:
                    with open('training_compd_headers.csv', 'w') as f:
                        df2.to_csv(f, index=False)
                else:
                    with open('training_compd_headers.csv', 'a') as f:
                        df2.to_csv(f, index=False, header=None)"""
                file_count += 1

            except UnicodeDecodeError:
                pass
            except TypeError:
                pass
            except pd.errors.ParserError:
                pass

"""
text2 = ['Effective Date;--(New Header)\nJob Number;--(New Header)\nJob Title;--(New Header)\nJob Description;--(New Header)]\n', 'other name', 'THESE HEADERS ARE COMPLETELY NEW AND SHOULDNT HAVE PREVIOUS OVERLAP.']
vector = vectorizer.transform(text2)
print(vector.toarray())
print(vector.toarray())
print(vector.toarray())

predicted2 = model2.predict(vector.toarray())

print(predicted2)

"""


