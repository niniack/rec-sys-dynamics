"""Cosin Similarity Class

This module provides recommendations given a dense ratings matrix.

To check out how to use this class, please refer to https://github.com/niniack/rec-sys-dynamics/blob/main/notebooks/rec-sys-movielens-data.ipynb

The required packages can be found in requirements.txt
"""

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


class CosinSimilarity(Recommender, Predictor):
    """
    Recommend new items by finding users that are the most similar to the given users using the cosin distance formula

    """

    def __init__(self, min_neighbors=1, min_sim=0, selector=None):

        # Set selector
        if selector is None:
            self.selector = basic.UnratedItemCandidateSelector()
        else:
            self.selector = selector

        # Set parameters
        self.min_neighbors = min_neighbors
        self.min_sim = min_sim

        # Enable logging
        _logger = logging.getLogger(__name__)

    def __str__(self):
        return "CosinSimilarity"

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

    # Add a user to the ratings matrix
    def add_user(self, user_id):

        # Check if user_id to be added already exists
        try:
            assert (
                user_id in self.user_index_
            ) == False, "User ID already exists! Not adding anything..."

        except AssertionError as e:
            print(e)
            exit(1)

        # Build a sparse matrix of length of number of items
        tmp_sparse_row = sparse.csr_matrix(np.zeros((1, len(self.item_index_))))

        # Vertically stack temporary matrix to original matrix
        self.rating_matrix_ = sparse.vstack([self.rating_matrix_, tmp_sparse_row])

        # Update user index
        self.user_index_ = self.user_index_.append(pd.Index([user_id]))

    # Add a user to the ratings matrix
    def add_item(self, item_id):

        # Check if item_id to be added already exists
        try:
            assert (item_id in self.item_index_) == False, "Item ID already exists!"

        except AssertionError as e:
            print(e)
            exit(1)

        # Build a sparse matrix of length of number of users
        tmp_sparse_col = sparse.csr_matrix(np.zeros((len(self.user_index_), 1)))

        # Horizotnally stack temporary matrix to original matrix
        self.rating_matrix_ = sparse.hstack([self.rating_matrix_, tmp_sparse_col])

        # Update item index
        self.item_index_ = self.item_index_.append(pd.Index([item_id]))

    # Add a user-item interaction for existing users and items
    def add_interactions(self, user_id, item_id, rating):

        # Check if inputs are lists and all input list lengths are equal
        assert type(user_id) == list, "Input user_id is not a list"
        assert type(item_id) == list, "Input item_id is not a list"
        assert type(rating) == list, "Input rating is not a list"
        assert (
            len(user_id) == len(item_id) == len(rating)
        ), "Input lists are not of the same length"

        # Build a temporary sparse LIL matrix

        tmp_ratings = sparse.lil_matrix(self.rating_matrix_.shape)

        for i in range(len(user_id)):

            # Obtain locations from ID
            (user_pos,) = np.where(self.user_index_ == user_id[i])[0]
            (item_pos,) = np.where(self.item_index_ == item_id[i])[0]

            # Fill into temporary sparse matrix
            tmp_ratings[user_pos, item_pos] = rating[i]

        # Convert temporary LIL to CSR
        tmp_ratings = tmp_ratings.tocsr()

        # Add temporary CSR to main ratings matrix
        self.rating_matrix_ += tmp_ratings

    # Provide a recommendation of top "n" movies given "user"
    # The recommender uses the UnratedItemCandidateSelector by default and uses the ratings matrix
    # it was originally fit on
    def recommend(self, user_id, candidates=None, ratings=None):

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
            )

        # minmax scale the popularity of each item
        normalized_popularity = np.interp(
            item_popularity, (item_popularity.min(), item_popularity.max()), (0, +1)
        )
        score = np.multiply(normalized_popularity, predicted_ratings)

        results = {
            "predicted_ratings": predicted_ratings,
            "normalized_popularity": normalized_popularity,
        }
        return pd.DataFrame(results, index=items)

        return None
