import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#movies_metadata = pd.read_csv('movies_metadata.csv')

data = {'movie_id': [1, 2, 3, 4],
        'title': ['The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump'],
        'genres': ['Crime, Drama', 'Action, Crime, Drama', 'Crime, Drama', 'Drama, Romance'],
        'description': ['The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
                        'When the menace known as the Joker emerges, he wreaks havoc and chaos on Gotham City.',
                        'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.',
                        'Forrest Gump, a man with low IQ, recounts the story of his life.'],
        'popularity': [9.8, 9.7, 9.6, 9.5]}

movies_df = pd.DataFrame(data)
#cleaning data, which we asume is already cleaned

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['description'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(movie_title, cosine_sim=cosine_sim):

    idx = movies_df.index[movies_df['title'] == movie_title].tolist()[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]  # Exclude the first one (the movie itself)

    movie_indices = [i[0] for i in sim_scores]
    top_movies = movies_df.iloc[movie_indices][['title', 'popularity']]

    top_movies['rank'] = top_movies['popularity'] * 0.5 + [sim[1] for sim in sim_scores] * 0.5
    top_movies = top_movies.sort_values(by='rank', ascending=False)
    
    return top_movies['title'].tolist()

recommendations = get_recommendations('The Godfather')
print("Top 30 Recommendations:", recommendations)
