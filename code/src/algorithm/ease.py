"""EASE Class

This module provides recommendations based on the EASE algorithm given a dense ratings matrix. For more on the EASE algorithm, see https://arxiv.org/abs/1905.03375

To check out how to use this class, please refer to 

Extends the SparseBasedAlgo class in algo.py

The required packages can be found in requirements.txt
"""

from .algo import SparseBasedAlgo

import pandas as pd
import numpy as np
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, Predictor, als, basic, user_knn
from scipy.sparse import csr_matrix, diags, linalg, hstack, vstack, lil_matrix
from scipy.linalg import inv
from lenskit.data import sparse_ratings

# Visualizations and debugging
import plotly.graph_objs as go
import logging


class EASE(SparseBasedAlgo):
    """
    Recommend new items by finding users that are the most similar to the given users using the EASE algorithm

    """

    def __init__(self, selector=None):
        # Set selector
        if selector is None:
            self.selector = basic.UnratedItemCandidateSelector()
        else:
            self.selector = selector

        # Enable logging
        _logger = logging.getLogger(__name__)

    def __str__(self):
        return "EASE"

    def get_num_users(self):
        return len(self.user_index_)

    def get_num_items(self):
        return len(self.item_index_)

    def fit(self, ratings, lambda_: float = 50, implicit=True):

        """
        ratings: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """

        matrix = ratings
        matrix["rating"] = 1

        # Get sparse representation in CSR format
        uir, users, items = sparse_ratings(matrix, scipy=True)

        # Store ratings
        self.rating_matrix_ = uir
        self.user_index_ = users
        self.item_index_ = items

        # Store lambda
        self.lambda_val = lambda_

        # Calculate score
        G = uir.transpose().dot(uir)
        lambda_diag = diags(np.full((G.shape[0]), lambda_))
        G += lambda_diag
        P = inv(G.toarray())
        B = P / (-np.diag(P))

        np.fill_diagonal(B, 0)
        self.B = B
        self.score = uir.dot(B)

        # Reduce candidate space to unseen items
        self.selector.fit(ratings)

    def update(self):

        # Recalculate score
        G = self.rating_matrix_.transpose().dot(self.rating_matrix_)
        lambda_diag = diags(np.full((G.shape[0]), self.lambda_val))
        G += lambda_diag
        P = inv(G.toarray())
        B = P / (-np.diag(P))

        np.fill_diagonal(B, 0)
        self.B = B
        self.score = self.rating_matrix_.dot(B)

        # Convert to dataframe for selector fit
        user_item_df = pd.DataFrame(
            data={
                "user": [self.user_index_[x] for x in self.rating_matrix_.nonzero()[0]],
                "item": [self.item_index_[x] for x in self.rating_matrix_.nonzero()[1]],
            }
        )

        self.selector.fit(user_item_df)
        # Update the item index on the candidate selector
        self.selector.items_ = self.item_index_
        # DON'T update the item index on the candidate selector
        # self.selector.users_ = self.user_index_

    def recommend(self, user_id, explore=False, candidates=None, ratings=None):

        # Reduce candidate space and store candidates with item ID
        if candidates is None:
            candidates = self.selector.candidates(user_id, ratings)

        (user_index,) = np.where(self.user_index_ == user_id)[0]

        # Predict ratings and scores for all unseen items
        prediction_score_df = self.predict_for_user(user_index, candidates)

        if explore:
            prediction_score_df = prediction_score_df[
                (prediction_score_df["normalized_popularity"] < 0.35)
            ]

        prediction_score_df = prediction_score_df.sort_values(
            by=["score"], ascending=False
        )

        return prediction_score_df

    def predict_for_user(self, user, items):

        # Instantiate ratings and item_popularity vectors
        item_popularity = np.zeros(len(items), dtype=float)

        # Grab item indices
        item_indices = []

        # Convert ratings matrix to COO matrix
        coo_ratings = self.rating_matrix_.tocoo()
        rating_matrix_items = coo_ratings.col

        # For each unseen item
        for i in range(len(items)):

            # Item position given item i ID
            item_pos = self.item_index_.get_loc(items[i])
            item_indices.append(item_pos)

            # Locations of ratings for item_pos
            (rating_locations,) = np.where(rating_matrix_items == item_pos)

            # Store popularity of item based on number of total ratings
            item_popularity[i] = len(rating_locations)

        # Grab the score vector for given user index
        all_scores = self.score[user]

        # Grab the unseen items
        unseen_item_scores = np.take(all_scores, item_indices)

        # minmax scale the popularity of each item
        normalized_popularity = np.interp(
            item_popularity, (item_popularity.min(), item_popularity.max()), (0, +1)
        )

        results = {
            "score": unseen_item_scores,
            "normalized_popularity": normalized_popularity,
        }
        return pd.DataFrame(results, index=items)
