import numpy as np
from numpy.linalg import norm

def cosineSimilarity(movieA, movieB):
    """
    Compute the cosine similarity between two movie vectors.

    Parameters:
        movieA (ndarray): Feature vector of the first movie.
        movieB (ndarray): Feature vector of the second movie.

    Returns:
        float: Cosine similarity score between the two movies.
    """
    dotProduct = np.dot(movieA, movieB)
    normA = norm(movieA)
    normB = norm(movieB)
    return dotProduct / (normA * normB)

def findSimilarMovies(targetMovie, movieList, topK=5):
    """
    Find the top-K most similar movies to the given target movie using cosine similarity.

    Parameters:
        targetMovie (ndarray): Feature vector of the target movie.
        movieList (list of ndarray): List of all other movie feature vectors.
        topK (int): Number of similar movies to return.

    Returns:
        list of tuple: A list of tuples (similarityScore, movieVector) for the top-K similar movies.
    """
    similarityScores = []
    for movie in movieList:
        similarity = cosineSimilarity(targetMovie, movie)
        similarityScores.append((similarity, movie))

    similarityScores.sort(reverse=True, key=lambda x: x[0])
    return similarityScores[:topK]

if __name__ == "__main__":
    # Example usage
    targetMovie = np.array([4.3, 323, 2023, 4.7, 13268])
    movieList = [
        np.array([4.7, 13268, 2023, 4.9, 1126]),
        np.array([5.0, 570, 2023, 4.8, 31813]),
        np.array([4.3, 7403, 2021, 4.1, 9259])
    ]

    similarMovies = findSimilarMovies(targetMovie, movieList)
    print(similarMovies)