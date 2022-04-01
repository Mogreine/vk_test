import os

import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split

from definitions import ROOT_DIR, DATA_DIR
from src.metrics import avg_precision_k, avg_precision
from src.models import ALS


def filter_data(ratings: pd.DataFrame, n_ratings_threshold: int = 3) -> pd.DataFrame:
    ratings_cnt = ratings.groupby("movieId").count().iloc[:, 0]
    ratings_cnt_mask = ratings_cnt >= n_ratings_threshold
    movies_filtered = set(ratings_cnt[ratings_cnt_mask].index)

    ratings = ratings[ratings["movieId"].isin(movies_filtered)]

    index_mapping = dict(zip(
        movies_filtered, range(len(movies_filtered))
    ))

    ratings["movieId"] = ratings["movieId"].map(lambda x: index_mapping[x])

    return ratings


def split_data(ratings: pd.DataFrame):
    ratings = ratings.sort_values(["timestamp"], ignore_index=True)

    train, test = train_test_split(ratings, train_size=0.7, shuffle=False)

    train_users = set(train["userId"].unique())
    test_users = set(test["userId"].unique())

    user_intersection = train_users.intersection(test_users)

    train = train[train["userId"].isin(user_intersection)]
    test = test[test["userId"].isin(user_intersection)]

    return train, test


def create_user_item(ratings: pd.DataFrame) -> sp.csr_matrix:
    implicit_ratings = ratings.loc[(ratings["rating"] >= 4)]

    users = implicit_ratings["userId"]
    movies = implicit_ratings["movieId"]

    user_item = sp.coo_matrix((np.ones_like(users), (users, movies))).tocsr()

    # confidence
    # user_item *= 40
    # user_item.data += 1

    return user_item


def aggr(ratings: pd.DataFrame) -> pd.Series:
    return ratings.groupby("userId")["movieId"].apply(list)


if __name__ == "__main__":
    ratings = pd.read_csv(os.path.join(DATA_DIR, "rating.csv"), nrows=None)

    ratings = filter_data(ratings)
    ratings_train, ratings_test = split_data(ratings)

    user_item = create_user_item(ratings_train)

    model = ALS(factors=64, iterations=5, calculate_training_loss=True, random_state=42)
    model.fit(user_item)

    train_ds = aggr(ratings_train)
    val_ds = aggr(ratings_test)

    precision_train = avg_precision_k(model, train_ds)
    precision_val = avg_precision_k(model, val_ds)

    print(f"Train precision: {precision_train: .3f}")
    print(f"Val precision: {precision_val: .3f}")

    print("Done!")
