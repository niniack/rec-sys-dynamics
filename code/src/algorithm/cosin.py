"""Cosin Similarity Class

This module provides recommendations based on cosin similarity given a dense ratings matrix.

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


class CosinSimilarity(SparseBasedAlgo):
    """
    Recommend new items by finding users that are the most similar to the given users using the cosin distance formula

    """

    def __init__(
        self, min_neighbors=1, min_sim=0, alpha=0.5, explore_percent=0.3, selector=None
    ):

        # Set selector
        if selector is None:
            self.selector = basic.UnratedItemCandidateSelector()
        else:
            self.selector = selector

        # Set parameters
        self.min_neighbors = min_neighbors
        self.min_sim = min_sim

        # Determines the weight given to normalized popularity
        self.alpha = alpha

        self.explore_percent = explore_percent

        # Enable logging
        _logger = logging.getLogger(__name__)

    def __str__(self):
        return "CosinSimilarity"

    def get_num_users(self):
        return len(self.user_index_)

    def get_num_items(self):
        return len(self.item_index_)

    # Store the ratings matrix in sparse format and generate similarity matrix
    def fit(self, ratings, **kwargs):

        # Get sparse representation in CSR format
        uir, users, items = sparse_ratings(ratings, scipy=True)

        # Store ratings
        self.rating_matrix_ = uir
        self.user_index_ = users
        self.item_index_ = items

        # Calculate similarites from sparse matrix
        self.sim_matrix_ = cosine_similarity(self.rating_matrix_)

        # Reduce candidate space to unseen items
        self.selector.fit(ratings)

    # Update the similarity matrix
    def update(self):

        # Calculate similarites from sparse matrix
        self.sim_matrix_ = cosine_similarity(self.rating_matrix_)

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

        # Grab similarity vector for user_index
        similarity_vector = self.sim_matrix_[0]

        # Predict ratings and scores for all unseen items
        prediction_score_df = self.predict_for_user(
            user_index, similarity_vector, candidates
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

    def predict_for_user(self, user, similarity_vector, items):

        # Instantiate ratings and item_popularity vectors
        predicted_ratings = np.zeros(len(items), dtype=float)
        item_popularity = np.zeros(len(items), dtype=float)

        # Convert ratings matrix to COO matrix
        coo_ratings = self.rating_matrix_.tocoo()
        rating_matrix_users = coo_ratings.row
        rating_matrix_items = coo_ratings.col
        rating_matrix_data = coo_ratings.data

        # For each unseen item
        for i in range(len(items)):

            # Item position given item i ID
            item_pos = self.item_index_.get_loc(items[i])

            # Locations of ratings for item_pos
            (rating_locations,) = np.where(rating_matrix_items == item_pos)

            # Store popularity of item based on number of total ratings
            item_popularity[i] = len(rating_locations)

            # Existing ratings by users
            i_ratings = rating_matrix_data[rating_locations]

            # Obtain user IDs from user positions
            i_raters_pos = rating_matrix_users[rating_locations]

            # Similarity values with all users that rated item i
            i_raters_similarities = similarity_vector[i_raters_pos]

            sum_of_product_of_similarities_and_ratings = np.multiply(
                i_raters_similarities, i_ratings
            ).sum()

            # Calculate floored ratings and scores
            sum_of_product_of_similarities_and_ratings = np.multiply(
                i_raters_similarities, i_ratings
            ).sum()
            sum_of_all_similarities = i_raters_similarities.sum()
            predicted_ratings[i] = (
                sum_of_product_of_similarities_and_ratings // sum_of_all_similarities
                if sum_of_all_similarities
                else 0
            )

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
