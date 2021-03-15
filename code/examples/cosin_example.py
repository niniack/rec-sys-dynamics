"""Example script to interact with the Cosin similarity class
"""
import sys
from pprintpp import pprint as prettyprint

### IMPORTANT ###
# Make sure you are correctly appending the path
# Otherwise the imports will not work!
sys.path.append("..")
from algorithm.cosin import CosinSimilarity

from lenskit.datasets import ML100K

movielens = ML100K("../../ml-100k")
ratings = movielens.ratings
ratings.head()

# Instantiate object
algo_cosin = CosinSimilarity()

# Reduce the candidates space + build user-user cosin similarity matrix
algo_cosin.fit(ratings)

# Obtain recommendations for user ID 1
recs = algo_cosin.recommend(1)

# Sort recommendations
recs = recs.sort_values(
    by=["predicted_ratings", "normalized_popularity"], ascending=False
)[["predicted_ratings", "normalized_popularity"]]

# Print the 30 top recommendations
prettyprint(recs.head(30))