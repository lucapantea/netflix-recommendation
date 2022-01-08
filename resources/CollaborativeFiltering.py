import numpy as np
from resources.BaselineEstimate import Baseline

class CFRecommender:
    MODEL = 'Collaborative Filtering'

    def __init__(self, utility_matrix, similarity_matrix,
                 ratings_per_userID, ratings_per_movie,
                 avg_user_ratings):
        self.utility_matrix = utility_matrix
        self.similarity_matrix = similarity_matrix
        self.ratings_per_userID = ratings_per_userID
        self.ratings_per_movie = ratings_per_movie
        self.avg_user_ratings = avg_user_ratings

    def collaborative_filtering(self, user_id_movie_id_pair):
        ratings_movie_j = self.utility_matrix[user_id_movie_id_pair[1]].values
        user_indices_that_have_rated_movie_j = (ratings_movie_j != 0).nonzero()
        user_i_sim = self.similarity_matrix[:, user_id_movie_id_pair[0] - 1][user_indices_that_have_rated_movie_j]
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

    def collaborative_filtering_with_baseline(self, user_id_movie_id_pair):
        ratings_movie_j = self.utility_matrix[user_id_movie_id_pair[1]].values
        user_indices_that_have_rated_first_movie = (ratings_movie_j != 0).nonzero()
        user_i_sim = self.similarity_matrix[:, user_id_movie_id_pair[0] - 1][user_indices_that_have_rated_first_movie]
        ratings_movie_j = ratings_movie_j[user_indices_that_have_rated_first_movie]
        ind = (np.logical_or(np.isnan(user_i_sim), user_i_sim == 1)).nonzero()
        sort_dec = (-user_i_sim).argsort()[:20]
        indices_nearest_neighbours = np.setdiff1d(sort_dec, ind)
        neighbourhood = user_i_sim[indices_nearest_neighbours]
        ratings_of_neighbours = ratings_movie_j[indices_nearest_neighbours]
        baseline_estimates_nn = np.array([])

        baseline = Baseline(self.ratings_per_userID, self.ratings_per_movie, self.avg_user_ratings)

        for i in indices_nearest_neighbours:
            # get global baseline estimates
            global_baseline_estimate = baseline.global_baseline_estimate(i + 1, user_id_movie_id_pair[1])
            baseline_estimates_nn = np.append(baseline_estimates_nn, global_baseline_estimate)

        global_estimate = baseline.global_baseline_estimate(user_id_movie_id_pair[0], user_id_movie_id_pair[1])
        predicted_rating = global_estimate + np.dot(neighbourhood,
                                                    np.subtract(ratings_of_neighbours, baseline_estimates_nn)) / np.sum(neighbourhood)
        if np.isnan(predicted_rating):
            predicted_rating = 3.0
        return predicted_rating

    def get_model_name(self):
        return self.MODEL_NAME
