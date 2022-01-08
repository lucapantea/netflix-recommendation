import numpy as np
import pandas as pd
import time


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


def RMSE(pred, actual):
    """
    Function that calculates the root mean square error.
    :param pred: the predicted value
    :param actual: the actual value
    :return: the root mean squared error
    """
    return np.sqrt(((pred - actual) ** 2).mean())


def clamp_rating(predicted_rating):
    if np.isnan(predicted_rating):
        predicted_rating = 3.0
    elif predicted_rating > 5.0:
        predicted_rating = 5.0
    elif predicted_rating < 1:
        predicted_rating = 1.0
    return predicted_rating


def add_ratings_for_movies_with_no_ratings():
    """
    As the title suggests, this method adds new ratings
    to movies that have not been rated by a user
    :return: the new ratings to be added
    """
    avg_user_ratings = np.genfromtxt('./resources/average_users.csv', delimiter=',')
    combined_movies_data_all_movies = pd.merge(movies_description, ratings_description, how='left', on='movieID')
    combined_movies_data_all_movies_numpy = combined_movies_data_all_movies.to_numpy()
    ratings_numpy = ratings_description.to_numpy()
    row_has_NaN = combined_movies_data_all_movies.isnull().any(axis=1)
    a = row_has_NaN.to_numpy()
    indices_of_nan = (a == True).nonzero()
    movies_with_no_ratings = combined_movies_data_all_movies_numpy[indices_of_nan[0]][:, 0]
    result = np.array(ratings_numpy, copy=True)
    temp = np.empty((0, 3))
    for i in range(1, 101):
        avg = avg_user_ratings[i - 1]
        for j in movies_with_no_ratings:
            temp = np.vstack([temp, [int(i), int(j), np.round(avg, decimals=4)]])
    result = np.concatenate((result, temp), axis=0)
    return result


np.seterr(divide='ignore', invalid='ignore')
predictions_numpy = predictions_description.to_numpy()

global_start = time.time()

ratings_numpy_all_movies = add_ratings_for_movies_with_no_ratings()

ratings_description_all_movies = pd.DataFrame.from_records(ratings_numpy_all_movies)
ratings_description_all_movies = ratings_description_all_movies.rename(columns={0: "userID", 1: "movieID", 2: "rating"})

combined_movies_data = pd.merge(movies_description, ratings_description, on='movieID')

start_time = time.time()
utility_matrix = ratings_description_all_movies.pivot_table(values='rating', index='userID', columns='movieID',
                                                            fill_value=0)


def predict_SGD(predictions_tuples):
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
        next_prediction_lf = np.dot(P[movieID - 1, :], Q[:, userID - 1])
        next_prediction_lf = clamp_rating(next_prediction_lf)
        result.append([idx, next_prediction_lf])
    return result


# -------------- SAVE RESULTS ---------------
predictions = predict_SGD(predictions_description)
print(predictions[0:10])

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formats data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)
    submission_writer.write(predictions)

print('Program finished.')
print('Total time elapsed: {:.2f}s'.format(time.time() - global_start))
