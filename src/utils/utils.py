import sys

import numpy as np
from random import randrange


def RMSE(pred, actual):
    return np.sqrt(((pred - actual) ** 2).mean())

def average_rating(user_id, ratings_per_userID):
    return np.mean(ratings_per_userID[user_id - 1])

def train_test_split(data_set, split=0.8, random_state=42):
    train_set = data_set.sample(frac=split, random_state=random_state)  # random state is a seed value
    test_set = data_set.drop(train_set.index)
    return np.array(train_set), np.array(test_set)

# rating values below 1 or above 5 do not make sense
def clamp_rating(predicted_rating):
    if np.isnan(predicted_rating):
        predicted_rating = 3.0
    elif predicted_rating > 5.0:
        predicted_rating = 5.0
    elif predicted_rating < 1:
        predicted_rating = 1.0
    return predicted_rating

def progress_bar(i, num_iter):
    sys.stdout.write('\r')
    sys.stdout.write("[{:{}}] {:.0f}%".format("=" * (i // (num_iter // 10)), 10, (100 / (num_iter - 1) * i)))
    sys.stdout.flush()