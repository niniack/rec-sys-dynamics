"""Recommendation Algorithm Base Class

This module is a base class for algorithms using sparse matrices

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
from abc import ABCMeta, abstractmethod

# Visualizations and debugging
import plotly.graph_objs as go
import logging


class SparseBasedAlgo(Recommender, Predictor, metaclass=ABCMeta):
    # def __init__(self):
    #     pass

    @abstractmethod
    def __str__(self):
        """Name of the class"""

    @abstractmethod
    def get_num_users(self):
        """Get the number of users in the recommender system"""

    @abstractmethod
    def get_num_items(self):
        """Get the number of items in the recommender system"""

    @abstractmethod
    def fit(self, ratings, **kwargs):
        """Fit the algorithm over the initial dataset

        :param ratings: user item ratings in a dataframe
        :type ratings: pandas DataFrame
        """

    @abstractmethod
    def update(self):
        """Refit the algorithm over the internal sparse matrix"""

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

    # Add an item to the ratings matrix
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
        self.rating_matrix_ = sparse.hstack(
            [self.rating_matrix_, tmp_sparse_col]
        ).tocsr()

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

            # If rating does not exist
            if self.rating_matrix_[user_pos, item_pos] == 0:
                # Fill into temporary sparse matrix
                tmp_ratings[user_pos, item_pos] = rating[i]

        # Convert temporary LIL to CSR
        tmp_ratings = tmp_ratings.tocsr()

        # Add temporary CSR to main ratings matrix
        self.rating_matrix_ += tmp_ratings