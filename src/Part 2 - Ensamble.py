import numpy as np
import pandas as pd
from random import randint
import time
import sys

# Where data is located
movies_file = 'data/movies.csv'
users_file = 'data/users.csv'
ratings_file = 'data/ratings.csv'
predictions_file = 'data/predictions.csv'
submission_file = 'data/submission.csv'

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID': 'int', 'year': 'int', 'movie': 'str'},
                                 names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';',
                                dtype={'userID': 'int', 'gender': 'str', 'age': 'int', 'profession': 'int'},
                                names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';',
                                  dtype={'userID': 'int', 'movieID': 'int', 'rating': 'int'},
                                  names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)


def average_rating(user_id):
    """
    Function that returns the average rating for a given user.
    :param user_id The id of the user
    :return the average rating for the given user
    """
    return np.mean(ratings_per_userID[user_id - 1])


def average_rating_movie(movie_id):
    """
    Function that returns the average rating for the movie given
    :param movie_id the id of the movies
    :return: the average rating for the given movies
    """
    a = ratings_per_movie.loc[ratings_per_movie['movieID'] == movie_id]
    # if no user has rated this movie return 0
    # this will mean that we do not add any bias for the global
    # baseline estimate for this movie
    if len(a['mean'].values) == 0:
        return 0
    else:
        return a['mean'].values[0]


def global_baseline_estimate(user_id, movie_id):
    """
    Function that returns the baseline estimated rating for a given
    movie and user
    :param user_id: the id of the user
    :param movie_id: the id of the movie
    :return: the global baseline estimate
    """
    overall_mean_rating = average_all_movies_all_users
    mean_of_user = avg_user_ratings[user_id - 1]
    mean_of_movie = average_rating_movie(movie_id)
    deviation_user = mean_of_user - overall_mean_rating
    deviation_movie = mean_of_movie - overall_mean_rating
    result = overall_mean_rating + deviation_user + deviation_movie
    return result


def progress_bar(i, num_iter):
    sys.stdout.write('\r')
    sys.stdout.write("[{:{}}] {:.0f}%".format("=" * (i // (num_iter // 10)), 10, (100 / (num_iter - 1) * i)))
    sys.stdout.flush()


# rating values below 1 or above 5 do not make sense
def clamp_rating(predicted_rating):
    if np.isnan(predicted_rating):
        predicted_rating = 3.0
    elif predicted_rating > 5.0:
        predicted_rating = 5.0
    elif predicted_rating < 1:
        predicted_rating = 1.0
    return predicted_rating


def cosine_similarity(v1, v2):
    """
    Function that returns the cosine similarity (Pearson correlation).
    :param v1: the first vector
    :param v2: the second vector
    :return: the Pearson correlation between the two vectors
    """
    return np.corrcoef(v1, v2)[1, 0]


def RMSE(pred, actual):
    """
    Function that calculates the root mean square error.
    :param pred: the predicted value
    :param actual: the actual value
    :return: the root mean squared error
    """
    return np.sqrt(((pred - actual) ** 2).mean())


def print_statistics():
    average_num_ratings_per_user = np.sum(utility_matrix_np != 0, axis=1).mean()
    num_ratings_per_user = np.sum(utility_matrix_np != 0, axis=1)
    idx_max_num_ratings_of_a_user = np.argmax(num_ratings_per_user)
    idx_min_num_ratings_of_a_user = np.argmin(num_ratings_per_user)
    print('Minimum number of ratings of a user: ',
          (np.sum(utility_matrix_np != 0, axis=1))[idx_min_num_ratings_of_a_user],
          ' ( userID =', idx_min_num_ratings_of_a_user, ')')
    print('Maximum number of ratings of a user: ', np.max(np.sum(utility_matrix_np != 0, axis=1)), ' ( userID =',
          idx_max_num_ratings_of_a_user, ')')
    print('Average number of ratings per user: ', average_num_ratings_per_user)
    print('Number of movies that have been rated by at least one user: ', combined_movies_data['movieID'].nunique())
    print('Average rating over all movies and all users', average_all_movies_all_users)


def add_ratings_for_movies_with_no_ratings():
    """
    As the title suggests, this method adds new ratings
    to movies that have not been rated by a user
    :return: the new ratings to be added
    """
    row_has_NaN = combined_movies_data_all_movies.isnull().any(axis=1)
    a = row_has_NaN.to_numpy()
    indices_of_nan = (a == True).nonzero()
    movies_with_no_ratings = combined_movies_data_all_movies_numpy[indices_of_nan[0]][:, 0]
    result = np.array(ratings_numpy, copy=True)
    temp = np.empty((0, 3))
    for i in range(1, 101):
        avg = average_rating(i)
        for j in movies_with_no_ratings:
            temp = np.vstack([temp, [int(i), int(j), np.round(avg, decimals=4)]])
    result = np.concatenate((result, temp), axis=0)
    return result


average_all_movies_all_users = 3.58131489029763
np.seterr(divide='ignore', invalid='ignore')

global_start = time.time()

combined_movies_data_all_movies = pd.merge(movies_description, ratings_description, how='left', on='movieID')
combined_movies_data_all_movies_numpy = combined_movies_data_all_movies.to_numpy()

predictions_numpy = predictions_description.to_numpy()
ratings_numpy = ratings_description.to_numpy()

ratings_per_movie = ratings_description.groupby('movieID')['rating'].agg(['count', 'mean']).reset_index()
ratings_numpy = ratings_numpy[ratings_numpy[:, 0].argsort()]
ratings_per_userID = np.split(ratings_numpy[:, 2], np.unique(ratings_numpy[:, 0], return_index=True)[1][1:])

start_time_pre_processing = time.time()
# 11 movies are not rated by any user. For these 11 movies we added ratings from 100 users
# and we use the user's average rating
start_time = time.time()
ratings_numpy_all_movies = add_ratings_for_movies_with_no_ratings()
print("----------time elapsed for adding ratings to 11 movies for 100 users: {:.2f}s----------".format(
    time.time() - start_time))

ratings_description_all_movies = pd.DataFrame.from_records(ratings_numpy_all_movies)
ratings_description_all_movies = ratings_description_all_movies.rename(columns={0: "userID", 1: "movieID", 2: "rating"})

combined_movies_data = pd.merge(movies_description, ratings_description, on='movieID')

start_time = time.time()
utility_matrix = ratings_description_all_movies.pivot_table(values='rating', index='userID', columns='movieID',
                                                            fill_value=0)
print("----------time elapsed for creating the utility matrix: {:.2f}s----------".format(time.time() - start_time))
# Function that prints some statistics
# print_statistics()

start_time = time.time()
utility_matrix_np = utility_matrix.to_numpy()
print("----------time elapsed for converting to numpy: {:.2f}s----------".format(time.time() - start_time))

# this is used for item-based CF
avg_movies_ratings = np.array([])
for i in range(1, 3707):
    avg_movies_ratings = np.append(avg_movies_ratings, average_rating_movie(i))

utility_matrix_np_transpose = utility_matrix_np.T
utility_matrix_np_norm = np.subtract(utility_matrix_np_transpose.T, avg_movies_ratings,
                                     where=(np.array(utility_matrix_np_transpose.T) != 0)).T
similarity_item_item = np.corrcoef(utility_matrix_np_norm)

start_time = time.time()
avg_user_ratings = np.genfromtxt('./resources/average_users.csv', delimiter=',')
print("----------time elapsed for reading average ratings per user: {:.2f}s----------".format(time.time() - start_time))

start_time = time.time()
utility_matrix_np = np.subtract(utility_matrix_np.T, avg_user_ratings, where=(np.array(utility_matrix_np.T) != 0)).T
print("----------time elapsed for normalizing utility matrix: {:.2f}s----------".format(time.time() - start_time))

start_time = time.time()
similarity_matrix = np.corrcoef(utility_matrix_np)
print("----------time elapsed for creating similarity matrix: {:.2f}s----------".format(time.time() - start_time))
print('----------time elapsed for Pre-Processing: {:.2f}s----------'.format(time.time() - start_time_pre_processing),
      '\n')

# profiling
year_released = movies_description['year'].to_numpy()
year_released[year_released == 0] = 1997
min_year = min(year_released)
max_year = max(year_released)

user_profile = users_description['age'].to_numpy()
user_profile = user_profile.astype(int)
min_age = min(user_profile)
max_age = max(user_profile)


def age_similarity(id_1, id_2):
    return 1 - abs(user_profile[id_1 - 1] - user_profile[id_2 - 1]) / (max_age - min_age)


def year_similarity(id_1, id_2):
    return 1 - abs(year_released[id_1 - 1] - year_released[id_2 - 1]) / (max_year - min_year)


"""
IMDb dataset integration.
Currently, the IMDb extracted columns will
aid us in developing a better understanding about
the user's preferences by adding the genre of movies
so our model can learn a user's preferences based on his/her
ratings.

@:param all_movies_encoded (3706, 27) contains all the necessary information
we need for further processing.
"""
# Where data is located
movies_file = 'imdb/IMDb movies.csv'
names_file = 'imdb/IMDb names.csv'
ratings_file = 'imdb/IMDb ratings.csv'

# Load data from a CSV file into pandas DataFrame
data_movies = pd.read_csv(movies_file)
data_names = pd.read_csv(names_file)
data_ratings = pd.read_csv(ratings_file)

movies = data_movies.copy()
names = data_names.copy()
ratings = data_ratings.copy()

duplicateRowsDF = movies_description[movies_description.duplicated(['movie'])].sort_values(by=['movie'])

# Replacing our current dataset with normal titles that can be identified with the ones from the IMDb dataset
movies_refactored = movies_description.copy()
movies_refactored['movie'] = movies_refactored['movie'].str.replace('_', ' ')
movies_refactored['movie'] = movies_refactored['movie'].str.replace(r"\(.*\)", "")
movies_refactored['movie'] = movies_refactored['movie'].str.strip()
all_movies = pd.merge(movies_refactored, movies, how='left', left_on='movie', right_on='original_title')

all_movies = pd.DataFrame(all_movies).reset_index()
given = pd.DataFrame(movies_refactored).reset_index()
imdb_given = all_movies.drop_duplicates('movieID', keep='first')

all_data_movies = imdb_given.copy()
print('All columns of the dataset:', list(all_data_movies.columns), '\n')
all_data_movies.drop(["index", "title", "original_title", "year_y", "date_published",
                      "language", 'country', 'language', 'director', 'writer',
                      'production_company', 'actors', 'description', 'budget',
                      'votes', 'reviews_from_users', 'reviews_from_critics',
                      'usa_gross_income', 'duration', 'worlwide_gross_income'], inplace=True, axis=1)

all_data_movies.rename(columns={'year_x': 'year'}, inplace=True)
print('All columns of the dataset, after removal:', list(all_data_movies.columns))

nan_values = np.where(all_data_movies != all_data_movies)
print('Total number of rows with missing data:', np.unique(nan_values).shape[0])
# Replace missing numerical values with the median of the column
# Replace missing string values with the most frequent value in the column
all_data_movies.fillna(all_data_movies.mean(), inplace=True)
all_data_movies.fillna(all_data_movies.mode(numeric_only=False).iloc[0], inplace=True)

# mapping to 1-5 scale
all_data_movies['avg_vote'] = all_data_movies['avg_vote'].apply(lambda x: (x * 5) / 10)
all_data_movies['metascore'] = all_data_movies['metascore'].apply(lambda x: (x * 5) / 100)

all_data_movies = all_data_movies[['movieID', 'imdb_title_id', 'year', 'movie', 'avg_vote', 'metascore', 'genre']]
all_data_movies.head()

# And finally, encoding
all_movies_encoded = pd.concat([all_data_movies.drop('genre', 1), all_data_movies['genre'].str.get_dummies(sep=", ")],
                               1)
all_movies_encoded.head()

# Now all numerical
all_movies_encoded = pd.concat([all_data_movies.drop('genre', 1), all_data_movies['genre'].str.get_dummies(sep=", ")],
                               1)
full_movies = all_movies_encoded.copy()
full_movies.drop(['avg_vote', 'metascore', 'year', 'movieID', 'imdb_title_id', 'movie'], inplace=True, axis=1)
full_movies.head()

from scipy.spatial import distance

full_movies_numpy = full_movies.to_numpy()
sim_version2 = np.corrcoef(full_movies_numpy)

print('Working on generating predictions...', '\n')

"""
Algorithms
"""


def collaborative_filtering_generic(user_id_movie_id_pair, user_based, dem, baseline):
    predicted_rating = None
    if user_based:
        ratings_movie_j = utility_matrix[user_id_movie_id_pair[1]].values
        user_indices_that_have_rated_movie_j = (ratings_movie_j != 0).nonzero()
        user_i_sim = similarity_matrix[:, user_id_movie_id_pair[0] - 1][user_indices_that_have_rated_movie_j]
        ratings_movie_j = ratings_movie_j[user_indices_that_have_rated_movie_j]
        ind = (np.logical_or(np.isnan(user_i_sim), user_i_sim == 1)).nonzero()
        sort_dec = (-user_i_sim).argsort()[:10]
        indices_nearest_neighbours = np.setdiff1d(sort_dec, ind)
        neighbourhood = user_i_sim[indices_nearest_neighbours]
        ratings_of_neighbours = ratings_movie_j[indices_nearest_neighbours]
        if baseline:
            baseline_estimates_nn = np.array([])
            for i in indices_nearest_neighbours:
                baseline_estimates_nn = np.append(baseline_estimates_nn,
                                                  global_baseline_estimate(i + 1, user_id_movie_id_pair[1]))

            global_estimate = global_baseline_estimate(user_id_movie_id_pair[0], user_id_movie_id_pair[1])
            predicted_rating = global_estimate + np.dot(neighbourhood,
                                                        np.subtract(ratings_of_neighbours,
                                                                    baseline_estimates_nn)) / np.sum(neighbourhood)
        else:
            if dem:
                dem = np.array([])
                for k in indices_nearest_neighbours:
                    dem = np.append(dem, age_similarity(k + 1, user_id_movie_id_pair[0]))
                enhanced_sim = np.multiply(dem, neighbourhood)
                predicted_rating = np.dot(enhanced_sim, ratings_of_neighbours) / np.sum(np.abs(enhanced_sim))
            else:
                predicted_rating = np.dot(neighbourhood, ratings_of_neighbours) / np.sum(neighbourhood)
    else:  # item-based
        ratings_user_j = utility_matrix_np_transpose[:, user_id_movie_id_pair[0] - 1]
        movie_indices_that_were_rated_by_user = (ratings_user_j != 0).nonzero()
        movie_i_sim = similarity_item_item[:, user_id_movie_id_pair[1] - 1][movie_indices_that_were_rated_by_user]
        ratings_user_j = ratings_user_j[movie_indices_that_were_rated_by_user]
        ind = (np.logical_or(np.isnan(movie_i_sim), movie_i_sim == 1)).nonzero()
        sort_dec = (-movie_i_sim).argsort()[:10]
        indices_nearest_neighbours = np.setdiff1d(sort_dec, ind)
        neighbourhood = movie_i_sim[indices_nearest_neighbours]
        ratings_of_neighbours = ratings_user_j[indices_nearest_neighbours]
        if baseline:
            baseline_estimates_nn = np.array([])
            for i in indices_nearest_neighbours:
                baseline_estimates_nn = np.append(baseline_estimates_nn,
                                                  global_baseline_estimate(user_id_movie_id_pair[0], i + 1))

            global_estimate = global_baseline_estimate(user_id_movie_id_pair[0], user_id_movie_id_pair[1])
            predicted_rating = global_estimate + np.dot(neighbourhood,
                                                        np.subtract(ratings_of_neighbours,
                                                                    baseline_estimates_nn)) / np.sum(
                neighbourhood)
        else:
            if dem:
                dem = np.array([])
                for k in indices_nearest_neighbours:
                    dist = (21 - distance.hamming(full_movies_numpy[k, :],
                                                  full_movies_numpy[user_id_movie_id_pair[1] - 1, :])) / 21
                    dem = np.append(dem, dist)
                    # dem = np.append(dem, sim_version2[k, user_id_movie_id_pair[1] - 1])
                    # dem = np.append(dem, year_similarity(k + 1, user_id_movie_id_pair[1]))
                enhanced_sim = np.multiply(dem, neighbourhood)
                predicted_rating = np.dot(enhanced_sim, ratings_of_neighbours) / np.sum(np.abs(enhanced_sim))
            else:
                predicted_rating = np.dot(neighbourhood, ratings_of_neighbours) / np.sum(neighbourhood)
    if np.isnan(predicted_rating):
        predicted_rating = 3.0
    return predicted_rating


def predict_cf(predictions_tuples, user_based, dem, baseline):
    start_t = time.time()
    number_predictions = len(predictions_tuples)
    result = []
    for idx in range(1, number_predictions + 1):
        next_prediction = collaborative_filtering_generic(
            (predictions_numpy[idx - 1][0], predictions_numpy[idx - 1][1]), user_based,
            dem, baseline)
        result.append([idx, next_prediction])
        progress_bar(idx, number_predictions)
    print("----------time elapsed for making predictions: {:.2f}min----------".format(
        (time.time() - start_t) / 60))
    return result


def predict_latent_factors_with_cf(predictions_tuples, user_based, dem, baseline):
    # Important to note that we user the avg of cb and lf here
    start = time.time()
    k = 40
    un, sn, vtn = np.linalg.svd(utility_matrix_np)
    normalized_pred = un[:, :k].dot(np.diag(sn[:k])).dot(vtn[:k])
    result_latent_factors = (normalized_pred.T + avg_user_ratings).T

    number_predictions = len(predictions_tuples)
    result = []
    for idx in range(1, number_predictions + 1):
        user_id = predictions_numpy[idx - 1][0]
        movie_id = predictions_numpy[idx - 1][1]
        next_prediction_lf = result_latent_factors[user_id - 1, movie_id - 1]
        next_prediction_cb = collaborative_filtering_generic((user_id, movie_id),
                                                             user_based, dem, baseline)
        result.append([idx, (next_prediction_cb + next_prediction_lf) / 2])
        progress_bar(idx, number_predictions)
    print("----------time elapsed for latent factors with CF: {:.2f}min----------".format((time.time() - start) / 60))
    return result


# if hybrid is true, the sgd prediction is combined with the item-based CF prediction
def predict_SGD(predictions_tuples, hybrid):
    k = 40
    R = utility_matrix.to_numpy().T
    P, S, Q = np.linalg.svd(R)
    P = P[:, :k]
    Q = Q[:k, :]
    n_epochs = 15
    for w in range(n_epochs):
        print(w)
        for i in range(3706):
            for x in range(6040):
                if R[i, x] != 0:
                    e_xi = 2 * (R[i, x] - np.dot(Q[:, x], P[i, :]))
                    Q[:, x] = Q[:, x] + 0.005 * (e_xi * P[i, :] - 0.07 * Q[:, x])
                    P[i, :] = P[i, :] + 0.005 * (e_xi * Q[:, x] - 0.07 * P[i, :])

    number_predictions = len(predictions_tuples)
    result = []
    for idx in range(1, number_predictions + 1):
        movieID = predictions_numpy[idx - 1][1]
        userID = predictions_numpy[idx - 1][0]
        next_pred = None
        if hybrid:
            sgd_pred = np.dot(P[movieID - 1, :], Q[:, userID - 1])
            i_cf_pred = collaborative_filtering_generic((userID, movieID), False, True, False)
            next_pred = 0.35 * i_cf_pred + 0.65 * sgd_pred
        else:
            next_pred = np.dot(P[movieID - 1, :], Q[:, userID - 1])
        next_pred = clamp_rating(next_pred)
        result.append([idx, next_pred])
    return result


def predict_random(predictions_tuples):
    number_predictions = len(predictions_tuples)
    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


# -------------- SAVE RESULTS ---------------
predictions = predict_SGD(predictions_description, True)
print(predictions[:10])

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formats data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)
    submission_writer.write(predictions)

print('Program finished.')
print('Total time elapsed: {:.2f}s'.format(time.time() - global_start))
