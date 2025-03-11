import pandas as pd

def loadData(moviesPath):
    """
    Load movie data from a CSV file.

    Parameters:
        moviesPath (str): Path to the CSV file containing movie data.

    Returns:
        DataFrame: Raw movie data loaded into a pandas DataFrame.
    """
    movies = pd.read_csv(moviesPath)
    return movies

def preprocessData(movies):
    """
    Clean and preprocess the movie dataset.

    Steps:
    - Remove rows with missing values.
    - Drop duplicate entries based on the movie title.
    - Normalize the 'Movie_Rating' column using standard scaling.
    - One-hot encode categorical columns: 'Format', 'MPAA_Rating', 'Directed_By', 'Starring'.
    - Convert 'ReleaseYear' column to integer type.

    Parameters:
        movies (DataFrame): Raw movie data.

    Returns:
        DataFrame: Preprocessed movie data ready for analysis or modeling.
    """
    # Remove rows with missing values
    movies.dropna(inplace=True)

    # Remove duplicate movies based on title
    movies.drop_duplicates(subset='title', inplace=True)

    # Normalize movie ratings
    movies['Movie_Rating'] = (
        (movies['Movie_Rating'] - movies['Movie_Rating'].mean()) /
        movies['Movie_Rating'].std()
    )

    # One-hot encode categorical features
    movies = pd.get_dummies(movies, columns=['Format', 'MPAA_Rating', 'Directed_By', 'Starring'])

    # Convert ReleaseYear to integer
    movies['ReleaseYear'] = movies['ReleaseYear'].astype(int)

    return movies

if __name__ == '__main__':
    moviesPath = '../data/Amazon- Movies and Films.csv'
    movies = loadData(moviesPath)
    movies = preprocessData(movies)
    print(movies)