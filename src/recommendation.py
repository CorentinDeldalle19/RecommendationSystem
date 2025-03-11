import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.data_preprocessing import preprocessData

def recommendMovies(movieIndex, similarityMatrix, movies, topN=3):
    """
    Recommend movies similar to a given movie using a cosine similarity matrix.

    Parameters:
        movieIndex (int): Index of the reference movie in the dataset.
        similarityMatrix (ndarray): Cosine similarity matrix.
        movies (DataFrame): Original movie DataFrame.
        topN (int): Number of similar movies to recommend.

    Returns:
        DataFrame: Top-N most similar movies.
    """
    similarIndices = similarityMatrix[movieIndex].argsort()[::-1][1:topN+1]
    similarMovies = movies.iloc[similarIndices]
    return similarMovies

if __name__ == "__main__":
    moviesPath = '../data/Amazon- Movies and Films.csv'
    movies = pd.read_csv(moviesPath)
    movies = preprocessData(movies)
    features = movies.select_dtypes(include=[np.number])
    similarityMatrix = cosine_similarity(features)
    movieIndex = 8
    recommendedMovies = recommendMovies(movieIndex, similarityMatrix, movies)
    print(recommendedMovies[['title', 'Movie_Rating', 'No_of_Ratings', 'ReleaseYear']])