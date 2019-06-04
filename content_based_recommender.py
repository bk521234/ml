import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('./the-movies-dataset/movies_metadata.csv', low_memory=False)



#Print plot overviews of the first 5 movies.
print(metadata['overview'].head())
print(metadata.shape)

# DROPPING UNEEDED COLUMNS TO FREE UP MEMORY
metadata = metadata.drop(['adult', 'budget', 'homepage', 'runtime', 'status', 'spoken_languages', 'production_countries', 'production_companies'], axis=1)

# import TfidfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# define a TF-IDF Vectorizer Object. remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

# construct the required TF-IDF matrixx by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

print(tfidf_matrix.shape)

# import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
print(indices)

# function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # get the index of the movie that matches the title
    idx = indices[title]

    # get the pairwise similarity scores of all the movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

print(get_recommendations('The Dark Knight Rises'))
print(get_recommendations('The Godfather'))

credits = pd.read_csv('./the-movies-dataset/credits.csv')
keywords = pd.read_csv('./the-movies-dataset/keywords.csv')

# remove rows with bad IDs
metadata = metadata.drop([19730, 29503, 35587])

# convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# merge keywords and credits into your main metadata dataframe
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')
print(metadata.head(2))

m = metadata['vote_count'].quantile(0.90)
print(m)

# FREE UP MEMORY!!!!!
# FREE UP MEMORY!!!!!
# FREE UP MEMORY!!!!!
# FREE UP MEMORY!!!!!
metadata = metadata.copy().loc[metadata['vote_count'] >= m]
metadata = metadata.reset_index()
print(metadata.shape)


# parse the stringigied features into their correspinding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

import numpy as np

# get the director's name from the crew feature. if director is not in list return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # check if more than 3 elements exist. if yes, return only first three. if no, return full list
        if len(names) > 3:
            names = names[:3]
        return names
    # return empty list in case of missing/malformed data
    return []

# difine new director, cast, genres and keywords features that are in a suitable form.
metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# print the new fatures of the first 3 films
print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # check if director exists. if not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres']) #.replace(u'\xa0', u'')

metadata['soup'] = metadata.apply(create_soup, axis=1)

# import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])

# compute the cosine similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# reset index of your main dataframe and construct reverse mapping as before
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])

print(get_recommendations('The Dark Knight Rises', cosine_sim2))

print(get_recommendations('The Godfather', cosine_sim2))
