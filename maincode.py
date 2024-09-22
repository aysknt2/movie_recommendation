import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('movies.csv')

df['Genres'] = df['Genres'].fillna('')
df['Description'] = df['Description'].fillna('')

df['Combined_Features'] = df['Genres'] + ' ' + df['Description']

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['Combined_Features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(movie_title, cosine_sim=cosine_sim, df=df):

    idx = df[df['Title'].str.contains(movie_title, case=False)].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]  # Skip the first one as it is the same movie
    
    movie_indices = [i[0] for i in sim_scores]
    
    recommendations = df.iloc[movie_indices][['Title', 'Rating']].sort_values(by='Rating', ascending=False)
    
    return recommendations
#example
movie_title = "Inception"
top_movies = get_recommendations(movie_title)
print(top_movies)
