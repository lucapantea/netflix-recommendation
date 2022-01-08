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


start_time = time.time()
utility_matrix_np = utility_matrix.to_numpy()
print("----------time elapsed for converting to numpy: {:.2f}s----------".format(time.time() - start_time))

# Function that prints some statistics
print_statistics()

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

print('Working on generating predictions...', '\n')


def collaborative_filtering(user_id_movie_id_pair):
    ratings_movie_j = utility_matrix[user_id_movie_id_pair[1]].values
    user_indices_that_have_rated_movie_j = (ratings_movie_j != 0).nonzero()
    user_i_sim = similarity_matrix[:, user_id_movie_id_pair[0] - 1][user_indices_that_have_rated_movie_j]
    ratings_movie_j = ratings_movie_j[user_indices_that_have_rated_movie_j]
    ind = (np.logical_or(np.isnan(user_i_sim), user_i_sim == 1)).nonzero()
    sort_dec = (-user_i_sim).argsort()[:10]
    indices_nearest_neighbours = np.setdiff1d(sort_dec, ind)
    neighbourhood = user_i_sim[indices_nearest_neighbours]
    ratings_of_neighbours = ratings_movie_j[indices_nearest_neighbours]
    predicted_rating = np.dot(neighbourhood, ratings_of_neighbours) / np.sum(neighbourhood)
    if np.isnan(predicted_rating):
        predicted_rating = 3.0
    return predicted_rating


def predict_collaborative_filtering(predictions_tuples):
    start_t = time.time()
    number_predictions = len(predictions_tuples)
    result = []
    for idx in range(1, number_predictions + 1):
        next_prediction = collaborative_filtering((predictions_numpy[idx - 1][0], predictions_numpy[idx - 1][1]))
        result.append([idx, next_prediction])
        progress_bar(idx, number_predictions)
    print("----------time elapsed for collaborative filtering: {:.2f}min----------".format(
        (time.time() - start_t) / 60))
    return result


def predict_latent_factors(predictions_tuples):
    start = time.time()
    k = 150
    un, sn, vtn = np.linalg.svd(utility_matrix_np)
    normalized_pred = un[:, :k].dot(np.diag(sn[:k])).dot(vtn[:k])
    result_latent_factors = (normalized_pred.T + avg_user_ratings).T

    number_predictions = len(predictions_tuples)
    result = []
    for idx in range(1, number_predictions + 1):
        next_prediction_lf = result_latent_factors[predictions_numpy[idx - 1][0] - 1, predictions_numpy[idx - 1][1] - 1]
        result.append([idx, next_prediction_lf])
    print("----------time elapsed for latent factors model (using SVD): {:.2f}min----------".format((time.time() - start) / 60))
    return result


def predict_random(predictions_tuples):
    number_predictions = len(predictions_tuples)
    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


# -------------- SAVE RESULTS ---------------
predictions = predict_collaborative_filtering(predictions_description)
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
