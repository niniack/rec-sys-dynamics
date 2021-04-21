import unittest
import pandas as pd
import os
from algorithm.ease import EASE
from algorithm.cosin import CosinSimilarity
import pandas.testing as pd_testing

import Datasets.Small_Test_Dataset as dataset


def getRatingsData():
    return pd.read_parquet(os.path.dirname(dataset.__file__) + "/ratings.parquet.gzip")


class TestSum(unittest.TestCase):
    def assertDataframeEqual(self, a, b, msg):
        try:
            pd_testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

    def ratings_fit(self):
        ratings = getRatingsData()
        algo = EASE()
        algo.fit(ratings)
        return ratings, algo

    def test_class_name(self):
        """
        Test that class name is returned correctly
        """
        algo = EASE()
        self.assertEqual(algo.__str__(), "EASE")

    def test_fit(self):
        """
        Test that runs a fit
        """
        ratings, algo = self.ratings_fit()
        self.assertEqual(algo.rating_matrix_.getnnz(), len(ratings))

    def test_original_num_users(self):
        ratings, algo = self.ratings_fit()
        self.assertEqual(len(algo.user_index_), 200)

    def test_original_num_users(self):
        ratings, algo = self.ratings_fit()
        self.assertEqual(len(algo.user_index_), 200)
        self.assertEqual(algo.get_num_users(), 200)

    def test_original_num_items(self):
        ratings, algo = self.ratings_fit()
        self.assertEqual(len(algo.item_index_), 400)
        self.assertEqual(algo.get_num_items(), 400)

    def test_get_recommendation(self):
        ratings, algo = self.ratings_fit()

        # Recommender for user 1
        first_rec = algo.recommend(1, explore=False)
        second_rec = algo.recommend(1, explore=False)
        self.assertEqual(first_rec, second_rec)

    def test_add_interaction(self):
        ratings, algo = self.ratings_fit()

        # Recommend for user 1
        first_rec = algo.recommend(1, explore=False)

        # Add interaction
        algo.add_interactions([1], [1], [2])

        # Call update
        algo.update()

        # Recommend again for user 1
        second_rec = algo.recommend(1, explore=False)

        # The two recommendations should not be the same
        self.assertNotEqual(
            first_rec.sort_index().index[0], second_rec.sort_index().index[0]
        )

    def test_add_new_user(self):
        ratings, algo = self.ratings_fit()
        new_user_id = 500

        # Add user with ID 500
        algo.add_user(new_user_id)

        # Update
        algo.update()

        # Check if updated correctly
        self.assertEqual(len(algo.user_index_), 201)
        self.assertEqual(algo.get_num_users(), 201)

    def test_add_new_user_and_interact(self):
        ratings, algo = self.ratings_fit()
        new_user_id = 500

        # Add user with ID 500
        algo.add_user(new_user_id)

        # Update
        algo.update()

        # Recommend to new user
        first_rec = algo.recommend(new_user_id, explore=False)

        # Interact with top item
        top_recommended_item = first_rec.sort_index().index[0]
        rating = 3
        algo.add_interactions([new_user_id], [top_recommended_item], [rating])
        algo.update()

        # Recommend to new user again
        second_rec = algo.recommend(new_user_id, explore=False)

        self.assertNotEqual(
            first_rec.sort_index().index[0], second_rec.sort_index().index[0]
        )


if __name__ == "__main__":
    unittest.main()