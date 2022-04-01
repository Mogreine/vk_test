import abc
import implicit
import numpy as np


class RecommenderBase(abc.ABC):
    @abc.abstractmethod
    def recommend(self, users, n_recommendations: int = 10, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def fit(*args, **kwargs):
        raise NotImplementedError()


class ALS(RecommenderBase):
    def __init__(self, factors=64, iterations=30, regularization=0.1, calculate_training_loss=True, random_state=42):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            calculate_training_loss=calculate_training_loss,
            random_state=random_state,
        )
        self.user_item = None

    def recommend(self, users, n_recommendations: int = 10, **kwargs) -> np.ndarray:
        movies, relevancy = [], []
        for user_id in users:
            movie_id, rel = zip(
                *self.model.recommend(user_id, self.user_item, N=n_recommendations, filter_already_liked_items=True)
            )

            movies.append(movie_id)
            relevancy.append(rel)

        return np.array(movies), np.array(rel)

    def fit(self, user_item):
        self.user_item = user_item
        self.model.fit(user_item.transpose())
