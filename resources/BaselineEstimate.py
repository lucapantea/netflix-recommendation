import numpy as np


class Baseline:

    def __init__(self, ratings_per_userID, ratings_per_movie,
                 avg_user_ratings):
        self.ratings_per_userID = ratings_per_userID
        self.ratings_per_movie = ratings_per_movie
        self.avg_user_ratings = avg_user_ratings
        self.average_all_movies_all_users = 3.58131489029763

    def average_rating(self, user_id):
        """
        Function that returns the average rating for a given user.
        :param user_id The id of the user
        :return the average rating for the given user
        """
        return np.mean(self.ratings_per_userID[user_id - 1])

    def average_rating_movie(self, movie_id):
        """
        Function that returns the average rating for the movie given
        :param movie_id the id of the movies
        :return: the average rating for the given movies
        """
        a = self.ratings_per_movie.loc[self.ratings_per_movie['movieID'] == movie_id]
        # if no user has rated this movie return 0
        # this will mean that we do not add any bias for the global
        # baseline estimate for this movie
        if len(a['mean'].values) == 0:
            return 0
        else:
            return a['mean'].values[0]

    def global_baseline_estimate(self, user_id, movie_id):
        """
        Function that returns the baseline estimated rating for a given
        movie and user
        :param user_id: the id of the user
        :param movie_id: the id of the movie
        :return: the global baseline estimate
        """
        overall_mean_rating = self.average_all_movies_all_users
        mean_of_user = self.avg_user_ratings[user_id - 1]
        mean_of_movie = self.average_rating_movie(movie_id)
        deviation_user = mean_of_user - overall_mean_rating
        deviation_movie = mean_of_movie - overall_mean_rating
        result = overall_mean_rating + deviation_user + deviation_movie
        return result
