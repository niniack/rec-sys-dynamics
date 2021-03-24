"""Example script to interact with the Cosin similarity class
"""
import sys
import numpy as np
from pprintpp import pprint as prettyprint

### IMPORTANT ###
# Make sure you are correctly appending the path
# Otherwise the imports will not work!
sys.path.append("..")
from src.algorithm.cosin import CosinSimilarity

from lenskit.datasets import ML100K

movielens = ML100K("../../ml-100k")
ratings = movielens.ratings
ratings.head()

# Instantiate object
algo_cosin = CosinSimilarity()

# Reduce the candidates space + build user-user cosin similarity matrix
algo_cosin.fit(ratings)

num_users = algo_cosin.get_num_users()
num_items = algo_cosin.get_num_items()
print("Initial number of users: " + str(num_users))
print("Initial number of items: " + str(num_items))


# Run a fake simulation
simulation_steps = 5
interactions = [[], [], []]
first_time_step = True
for t in range(1, simulation_steps + 1):

    # Add a new user and item
    algo_cosin.add_user(num_users + t)
    algo_cosin.add_item(num_items + t)

    # Add two random interactions
    # np.random.randint(low is inclusive, high is exclusive)
    interactions = 2
    random_users = np.random.randint(1, algo_cosin.get_num_users(), interactions)
    random_items = np.random.randint(1, algo_cosin.get_num_items() + 1, interactions)
    random_ratings = np.random.randint(1, 6, interactions)

    algo_cosin.add_interactions(
        random_users.tolist(), random_items.tolist(), random_ratings.tolist()
    )

    # Refit
    algo_cosin.update()

    # Obtain recommendations for a new user
    random_user_for_recommendations = num_users + t
    recs = algo_cosin.recommend(random_user_for_recommendations)

    # Sort recommendations
    recs = recs.sort_values(
        by=["predicted_ratings", "normalized_popularity"], ascending=False
    )[["predicted_ratings", "normalized_popularity"]]

    # Print a few top recommendations
    print("Recommendations for User", random_user_for_recommendations, ": ")
    prettyprint(recs.tail(30))