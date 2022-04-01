import numpy as np
import pandas as pd

from src.models import RecommenderBase


def avg_precision_k(model: RecommenderBase, relevant_movies: pd.Series, k: int = 10):
    recommended_movies, _ = model.recommend(relevant_movies.index, k)
    res = 0
    for pred, target in zip(recommended_movies, relevant_movies):
        pred = set(pred)
        target = set(target)
        res += len(pred.intersection(target)) / len(pred)

    return res / len(relevant_movies)


def avg_precision(model: RecommenderBase, relevant_movies: pd.Series):
    res = 0
    for user_id in relevant_movies.index:
        recommended_movies, _ = model.recommend([user_id], len(relevant_movies[user_id]))
        recommended_movies = recommended_movies[0]
        res += len(set(recommended_movies).intersection(relevant_movies[user_id])) / len(recommended_movies)

    return res / len(relevant_movies)


def calc_auc(model, user_item):
    user_item = user_item.toarray()
    auc = 0
    for u_id, u in enumerate(user_item):
        if np.count_nonzero(u != 0) == 0:
            continue

        pos_mask = u > 0
        neg_mask = ~u

        preds = model.I @ model.U[u_id]
        comp_matrix = preds[pos_mask].reshape(-1, 1) > preds[neg_mask]
        u_auc = np.count_nonzero(comp_matrix) / (comp_matrix.shape[0] * comp_matrix.shape[1])

        auc += u_auc

    return auc / user_item.shape[0]
