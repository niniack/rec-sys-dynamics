"""Matrix Factorization Class

This module provides recommendations based on regularized nonnegative matrix factorization given a dense ratings matrix.

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
from numpy import linalg as LA
from numba import jit, njit, prange
import numba

# @numba.jit(nopython=True, parallel=True)
@jit(nopython=True)
# @njit(parallel=True)
def matrix_factorization_pred(X, P, Q, K, steps, alpha, beta, Mask):
    #    Mask = (X!=0)
    Q = Q.T
    error_list = np.zeros(steps)
    for step in range(steps):
        print(step)
        # for each user
        for i in prange(X.shape[0]):
            # for each item
            for j in range(X.shape[1]):
                if X[i, j] > 0:

                    # calculate the error of the element
                    eij = X[i, j] - np.dot(P[i, :], Q[:, j])
                    # second norm of P and Q for regularilization
                    sum_of_norms = 0
                    # for k in xrange(K):
                    #    sum_of_norms += LA.norm(P[:,k]) + LA.norm(Q[k,:])
                    # added regularized term to the error
                    sum_of_norms += LA.norm(P) + LA.norm(Q)
                    # print sum_of_norms
                    eij += (beta / 2) * sum_of_norms
                    # compute the gradient from the error
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (
                            2 * eij * Q[k][j] - (beta * P[i][k])
                        )
                        Q[k][j] = Q[k][j] + alpha * (
                            2 * eij * P[i][k] - (beta * Q[k][j])
                        )

        # compute total error
        error = 0
        # for each user
        extimated_X = np.trunc(P @ Q)
        extimated_X = np.where(extimated_X > 5, 5, extimated_X)
        extimated_X = np.where(extimated_X < 0, 0, extimated_X)
        extimated_error = np.multiply(X - extimated_X, Mask)
        error = LA.norm(extimated_error)
        error_list[step] = error

        if error < 0.001:
            break
    return extimated_X, P, Q.T, error_list


def matrix_factorization_pred_naive(X, P, Q, K, steps, alpha, beta, Mask):
    #    Mask = (X!=0)
    Q = Q.T
    error_list = np.zeros(steps)
    for step in range(steps):
        print(step)
        # for each user
        for i in range(X.shape[0]):
            # for each item
            for j in range(X.shape[1]):
                if X[i, j] > 0:

                    # calculate the error of the element
                    eij = X[i, j] - np.dot(P[i, :], Q[:, j])
                    # second norm of P and Q for regularilization
                    sum_of_norms = 0
                    # for k in xrange(K):
                    #    sum_of_norms += LA.norm(P[:,k]) + LA.norm(Q[k,:])
                    # added regularized term to the error
                    sum_of_norms += LA.norm(P) + LA.norm(Q)
                    # print sum_of_norms
                    eij += (beta / 2) * sum_of_norms
                    # compute the gradient from the error
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (
                            2 * eij * Q[k][j] - (beta * P[i][k])
                        )
                        Q[k][j] = Q[k][j] + alpha * (
                            2 * eij * P[i][k] - (beta * Q[k][j])
                        )

        # compute total error
        error = 0
        # for each user
        extimated_X = np.trunc(P @ Q)
        extimated_X = np.where(extimated_X > 5, 5, extimated_X)
        extimated_X = np.where(extimated_X < 0, 0, extimated_X)
        extimated_error = np.multiply(X - extimated_X, Mask)
        error = LA.norm(extimated_error)
        error_list[step] = error

        if error < 0.001:
            break
    return extimated_X, P, Q.T, error_list


class MatrixFactorization(SparseBasedAlgo):
    """
    Recommend new items by completing the user-item rating matrix with regularized nonnegative matrix factorization
    """

    def __init__(
        self, K=8, steps=100, alpha=0.0002, beta=float(0.02), gamma=0.5, selector=None
    ):

        # Set selector
        if selector is None:
            self.selector = basic.UnratedItemCandidateSelector()
        else:
            self.selector = selector

        # Set parameters
        self.steps = steps
        self.K = K
        self.alpha = alpha
        self.beta = beta

        # Determines the weight given to normalized popularity
        # This is the same parameter of alpha in cosin.py
        self.gamma = gamma

        # Enable logging
        _logger = logging.getLogger(__name__)

    def __str__(self):
        return "MatrixFactorization"

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
        # Calculate mask matrix from rating matrix
        self.mask_matrix = self.rating_matrix_ != 0

        # Calculate completed u-i matrix matrix with matrix factorization
        # Grab the input rating matrix and mask matrix
        M = self.rating_matrix_
        M_s = self.mask_matrix
        # P: an initial matrix of dimension N x K, where is n is no of users and k is hidden latent features
        P = np.random.rand(M.shape[0], self.K)
        # Q : an initial matrix of dimension M x K, where M is no of movies and K is hidden latent features
        Q = np.random.rand(M.shape[1], self.K)

        self.full_matrix_, _, _, _ = matrix_factorization_pred(
            M.todense(), P, Q, self.K, self.steps, self.alpha, self.beta, M_s.todense()
        )

        # Reduce candidate space to unseen items
        self.selector.fit(ratings)

    # Update the predicted u-i matrix
    def update(self):

        # Calculate mask matrix from rating matrix
        self.mask_matrix = self.rating_matrix_ != 0

        # Calculate completed u-i matrix matrix with matrix factorization
        # Grab the input rating matrix and mask matrix
        M = self.rating_matrix_
        M_s = self.mask_matrix
        # P: an initial matrix of dimension N x K, where is n is no of users and k is hidden latent features
        P = np.random.rand(M.shape[0], self.K)
        # Q : an initial matrix of dimension M x K, where M is no of movies and K is hidden latent features
        Q = np.random.rand(M.shape[1], self.K)

        self.full_matrix_, _, _, _ = matrix_factorization_pred(
            M.todense(), P, Q, self.K, self.steps, self.alpha, self.beta, M_s.todense()
        )

        # Convert to dataframe for selector fit

        user_item_df = pd.DataFrame(
            data={
                "user": [x + 1 for x in self.rating_matrix_.nonzero()[0]],
                "item": [x + 1 for x in self.rating_matrix_.nonzero()[1]],
            }
        )

        self.selector.fit(user_item_df)
        # Update the item index on the candidate selector
        self.selector.items_ = self.item_index_
        # DON'T update the item index on the candidate selector
        # self.selector.users_ = self.user_index_

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
            user_index, self.full_matrix_, candidates
        )

        if explore:
            prediction_score_df = prediction_score_df[
                (prediction_score_df["predicted_ratings"] == 0)
                | (prediction_score_df["normalized_popularity"] < 0.25)
            ]

        prediction_score_df = prediction_score_df.sort_values(
            by=["score"], ascending=False
        )

        return prediction_score_df

    def predict_for_user(self, user, extimated_X, items):

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

            # predicted_ratings[i] = extimated_X[user_pos,item_pos]
            predicted_ratings[i] = extimated_X[user, item_pos]

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
            self.gamma * normalized_popularity, (1 - self.gamma) * normalized_rating
        )

        results = {
            "predicted_ratings": predicted_ratings,
            "normalized_popularity": normalized_popularity,
            "score": score,
        }
        return pd.DataFrame(results, index=items)