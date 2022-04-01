import os

import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib import pyplot as plt

from definitions import ROOT_DIR, DATA_DIR


def analyze_movies(ratings: pd.DataFrame, movies: pd.DataFrame):
    cnt = ratings.groupby("movieId").count().iloc[:, 0]
    print(f"Median: {cnt.median()}")
    print(f"Max: {cnt.max()}")
    print(f"Min: {cnt.min()}")
    cnt[cnt < 20].plot(kind="hist", bins=np.arange(21), xticks=np.arange(20))

    plt.ylabel("Number of movies")
    plt.xlabel("Times rated")
    plt.show()

    print(np.sum(cnt[cnt < 3]) / len(movies["movieId"].unique()))


def analyze_users(ratings: pd.DataFrame):
    cnt = ratings.groupby("userId").count().iloc[:, 0]
    print(f"Median: {cnt.median()}")
    print(f"Max: {cnt.max()}")
    print(f"Min: {cnt.min()}")
    cnt[cnt < 100].plot(kind="hist", bins=np.arange(20, 100, 5), xticks=np.arange(20, 101, 5))

    plt.ylabel("Number of users")
    plt.xlabel("Times rated")
    plt.show()

    print(np.sum(cnt[cnt < 3]) / 138493)


if __name__ == "__main__":
    n_rows = None
    ratings = pd.read_csv(os.path.join(DATA_DIR, "rating.csv"), nrows=n_rows)
    movies = pd.read_csv(os.path.join(DATA_DIR, "movie.csv"), nrows=n_rows)

    # analyze_movies(ratings, movies)
    analyze_users(ratings)
