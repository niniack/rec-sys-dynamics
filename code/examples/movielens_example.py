"""Example script to interact with the movielens and data class
"""
import sys

### IMPORTANT ###
# Make sure you are correctly appending the path
# Otherwise the imports will not work!
sys.path.append("..")
from src.movielens_data.movielens import movielens
from src.movielens_data.movielens import cluster

from lenskit.datasets import ML100K

movielens = ML100K("../../ml-100k")
ratings = movielens.ratings
movies = movielens.movies

ratings.head()
movies.head()

# Instantiate movielens object
data = movielens(movies, ratings)

# Creation of different datasets for SVD (weighted, absolute)

# User-Item Interaction Matrix
dataUI = data.UserItem()

# SVD reduced User-Item Interaction Matrix

# enter number of latent features you want, and which dataset you want the SVD performed on (default is complete User-Movie matrix)
# UI = User-Item matrix
# GR = Average rating for each genre for each user
# wGR = Weighted genre ratings for each user
dataSVD_UI = data.SVDmatrix(3, dataset = 'UI)

# Instantiate clustering object
UI_clusters = cluster(dataSVD_UI)

### IMPORTANT: Make sure you train the model/clusters before plotting them. 
# perform GMM clustering to get cluster values and cluster probabilities
# df = 'pred' for cluster predictions, 'proba' for clusters + probabilities, 'full' for latent values+clusters+probas
cluster_probas = UI_clusters.gmm(n=3,covariance_type="full",df=proba)


# plot data and clusters (True for colour-coded clusters, if False it will just plot latent features in monotone)
# These functions currently only work in JupyterNotebook.                            
#UI_clusters.plotScatter(True, 'kmeans')
#UI_clusters.plotScatter(True, 'gmm')                            
                            