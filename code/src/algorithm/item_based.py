"""Item-Based Collaborative Filtering Class

This module provides recommendations based on Item-Based Collaborative Filtering.

To check out how to use this class, please refer to https://github.com/niniack/rec-sys-dynamics/blob/main/notebooks/rec-sys-movielens-data.ipynb

Extends the SparseBasedAlgo class in algo.py

The required packages can be found in requirements.txt
"""

from .algo import SparseBasedAlgo

from math import isnan
import pandas as pd
import numpy as np
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, Predictor, als, basic, user_knn
from lenskit.data import sparse_ratings
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# Visualizations and debugging
import plotly.graph_objs as go
import logging

from scipy.sparse.linalg import svds, eigs, norm
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class ItemBasedCosinSimilarity(Recommender, Predictor):
    """
    Recommend new items by finding items that are the most similar to already rated items by users

    """

    def __init__(self, n_neighbors=11, min_neighbors=1, min_sim=0, alpha=0.5, explore_percent=0.3, selector=None):

        # Set selector
        if selector is None:
            self.selector = basic.UnratedItemCandidateSelector()
        else:
            self.selector = selector

        # Set parameters
        self.min_neighbors = min_neighbors
        self.min_sim = min_sim
        self.n_neighbors = n_neighbors

        # Determines the weight given to normalized popularity
        self.alpha = alpha

        self.explore_percent = explore_percent

        # Enable logging
        _logger = logging.getLogger(__name__)

    def __str__(self):
        return 'ItemBasedCosinSimilarity'

    # Store the ratings matrix in sparse format and generate similarity matrix
    def fit(self, ratings, **kwargs):

        # Get sparse representation in CSR format
        uir, users, items = sparse_ratings(ratings, scipy=True)

        # Store ratings
        self.rating_matrix_ = uir
        self.user_index_ = users
        self.item_index_ = items
        self.rating_matrix_iur = self.rating_matrix_.T

        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(self.rating_matrix_iur)
        distances, indices = knn.kneighbors(self.rating_matrix_iur, n_neighbors=self.n_neighbors)

        self.distances = distances
        self.indices = indices

        # Reduce candidate space to unseen items
        self.selector.fit(ratings)

    # Update the similarity matrix
    def update(self):

        self.rating_matrix_iur = self.rating_matrix_.T

        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(self.rating_matrix_iur)
        distances, indices = knn.kneighbors(self.rating_matrix_iur, n_neighbors=self.n_neighbors)

        self.distances = distances
        self.indices = indices

        # Convert to dataframe for selector fit

        user_item_df = pd.DataFrame(
            data={
                "user": [self.user_index_[x] for x in self.rating_matrix_.nonzero()[0]],
                "item": [self.item_index_[x] for x in self.rating_matrix_.nonzero()[1]],
            }
        )

        self.selector.fit(user_item_df)
        # Update the item index on the candidate selector
        item_diff = self.item_index_.difference(self.selector.items_)
        self.selector.items_ = self.selector.items_.append(item_diff)

    # Provide a recommendation of top "n" movies given "user"
    # The recommender uses the UnratedItemCandidateSelector by default and uses the ratings matrix
    # it was originally fit on
    def recommend(self, user_id, explore=False, candidates=None, ratings=None):

        # Reduce candidate space and store candidates with item ID
        if candidates is None:
            candidates = self.selector.candidates(user_id, ratings)

        # Grab user index for given user_id
        (user_index,) = np.where(self.user_index_ == user_id)[0]

        # Predict ratings and scores for all unseen items
        prediction_score_df = self.predict_for_user(
            user_index, candidates
        )

        if explore:
            # calculate number of items based on percentage
            number_of_items = int(prediction_score_df.shape[0] * self.explore_percent)

            # Grab items that have a predicted rating of 0
            zero_predicted_ratings_df = prediction_score_df[
                (prediction_score_df["predicted_ratings"] == 0)
            ]

            # Sort by normalized popularity
            prediction_score_df = prediction_score_df.sort_values(
                by=["normalized_popularity"], ascending=True
            ).head(number_of_items)

            # Concatenate the two dfs
            prediction_score_df = pd.concat(
                [prediction_score_df, zero_predicted_ratings_df]
            )

            # Drop duplicates
            prediction_score_df = prediction_score_df[
                ~prediction_score_df.index.duplicated()
            ]

        prediction_score_df = prediction_score_df.sort_values(
            by=["score"], ascending=False
        )

        return prediction_score_df

    def predict_for_user(self, user, items):

        # Instantiate ratings and item_popularity vectors
        predicted_ratings = np.zeros(len(items), dtype=float)
        item_popularity = np.zeros(len(items), dtype=float)

        coo_ratings = self.rating_matrix_.tocoo()
        #rating_matrix_users = coo_ratings.row
        rating_matrix_items = coo_ratings.col
        #rating_matrix_data = coo_ratings.data

        iur = self.rating_matrix_iur
        #iur_copy = iur.copy()
        for i in range(len(items)):

            m = self.item_index_.get_loc(items[i])
            sim_movies = self.indices[m].tolist()
            movie_distances = self.distances[m].tolist()

            if m in sim_movies:
                id_movie = sim_movies.index(m)
                sim_movies.remove(m)
                movie_distances.pop(id_movie)

            else:
                sim_movies = sim_movies[:self.n_neighbors - 1]
                movie_distances = movie_distances[:self.n_neighbors - 1]

            # movie_similarty = 1 - movie_distance
            movie_similarity = [1 - x for x in movie_distances]
            movie_similarity_copy = movie_similarity.copy()
            nominator = 0

            # for each similar movie
            for s in range(0, len(movie_similarity)):

                # check if the rating of a similar movie is zero
                if iur[sim_movies[s], user] == 0:

                    # if the rating is zero, ignore the rating and the similarity in calculating the predicted rating
                    if len(movie_similarity_copy) == (self.n_neighbors - 1):
                        movie_similarity_copy.pop(s)

                    else:
                        movie_similarity_copy.pop(s - (len(movie_similarity) - len(movie_similarity_copy)))

                # if the rating is not zero, use the rating and similarity in the calculation
                else:
                    nominator = nominator + movie_similarity[s] * iur[sim_movies[s], user]

            # check if the number of the ratings with non-zero is positive
            if len(movie_similarity_copy) > 0:

                # check if the sum of the ratings of the similar movies is positive.
                if sum(movie_similarity_copy) > 0:
                    predicted_r = nominator / sum(movie_similarity_copy)

                else:
                    predicted_r = 0

            # if all the ratings of the similar movies are zero, then predicted rating should be zero
            else:
                predicted_r = 0

            predicted_ratings[i] = predicted_r

            # Item position given item i ID
            item_pos = self.item_index_.get_loc(items[i])

            # Locations of ratings for item_pos
            rating_locations, = np.where(rating_matrix_items == item_pos)

            # Store popularity of item based on number of total ratings
            item_popularity[i] = len(rating_locations)

        # minmax scale the popularity of each item
        normalized_popularity = np.interp(
            item_popularity, (item_popularity.min(), item_popularity.max()), (0, +1)
        )
        normalized_rating = np.interp(
            predicted_ratings,
            (predicted_ratings.min(), predicted_ratings.max()),
            (0, +1),
        )

        score = np.add(
            self.alpha * normalized_popularity, (1 - self.alpha) * normalized_rating
        )

        results = {
            "predicted_ratings": predicted_ratings,
            "normalized_popularity": normalized_popularity,
            "score": score,
        }
        return pd.DataFrame(results, index=items)
