"""Example script to interact with the movielens and data class
"""
import sys
import pandas as pd

### IMPORTANT ###
# Make sure you are correctly appending the path
# Otherwise the imports will not work!
sys.path.append("..")
from src.analysis.cluster import movielens, cluster

# Import Lenskit Dataset
from lenskit.datasets import ML100K

ML100K = ML100K("/Users/pvs262/Library/Python/3.8/lib/python/site-packages/lenskit/data/ml-100k")
ratings = ML100K.ratings
movies = ML100K.movies

# Instantiate MovieLens object
data = movielens(movies, ratings)
# Get User-Item Interaction Matrix from MovieLens object
dataUI = data.UserItem() 

# Instantiate Clustering object
ML_clusters = cluster(dataUI)

# One-time set up to choose the number of latent features we want. 
# If you want to change the number of latent features in the reduced dataset, you need to use cluster.SVD(n) again.
ML_clusters.svd(3)

# Function to add user/item with ID
ML_clusters.add_user(len(dataUI)+1) 
ML_clusters.add_item(len(dataUI.columns)+1)

# Function to add user/item automatically (it returns the ID of the new user/item)
ML_clusters.add_user_auto() 
ML_clusters.add_item_auto()

# Function to add ratings for EXISTING users and items where cluster.addRating(user_id, item_id, rating)
ML_clusters.add_interactions([4],[2],[3])

### IMPORTANT ### 
# Make sure you train the model/clusters before plotting them using cluster.gmm(n, covariance_type, df)m
# cluster.gmm(n, covariance_type, df) calculates SVD to find updated reduced dataset and then clusters
# returns cluster number for each user and the probability of belonging to each of the n clusters

probas = ML_clusters.gmm(n=3,covariance_type="full",df='proba')
print(probas)

