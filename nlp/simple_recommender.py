import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('./the-movies-dataset/movies_metadata.csv', low_memory=False)

print(metadata.head(3))

# calculate C
C = metadata['vote_average'].mean()
print(C)

m = metadata['vote_count'].quantile(0.90)
print(m)

q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
print(q_movies.shape)

# function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# define a new feature 'score' and calculate its value with 'weighted_rating()'
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

# sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

# print the top 15 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))
