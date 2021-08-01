"""Movielens Data Class and Cluster Class
This module processes the movielens dataset and provides clustering functions for a given dataset.
To check out how to use this class, please refer to https://github.com/niniack/rec-sys-dynamics/blob/input_pipeline/notebooks/movielens-data.ipynb
The required packages can be found in requirements.txt
"""

# Import the libraries we will be using
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (5, 3),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'font.size':8}
plt.rcParams.update(params)


import logging

from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, PCA

import plotly.offline as py
import plotly.graph_objs as go

# MOVIELENS DATASET CLASS - easy access functions to process movielens dataset
class movielens:
    
     # constructor taking in dataset (genre_ratings), number of maximum clusters
    def __init__(self, movies, ratings):
        
        # assign input rating matrix
        self.movies = movies.drop(columns = ['release','vidrelease','imdb'], axis = 1)
        self.ratings = ratings
        self.genres = self.movies.columns.tolist()[1:]
        
        # Enable logging
        self._logger = logging.getLogger(__name__)
        
    def __str__(self):
        return 'MovieLens Dataset'
    
    # Function to return list of strings of genres in MovieLens dataset
    def get_genres(self):
        return self.genres
        
    # Function to return dataframe of user (rows) and movie (columns) ratings - a user-item interaction matrix
    def UserItem(self):
        self.UI_matrix = self.ratings.merge(self.movies,on='item', how='left')
        self.UI_matrix = self.UI_matrix.pivot_table(index='user',columns='item',values='rating')
        self.UI_matrix = self.UI_matrix.fillna(0)
        return self.UI_matrix
    
    # Function to count total number of movies a user has rated
    def TotalUserRatings(self):
        total_user_ratings = self.ratings.groupby(['user']).count().drop(columns = ['item','timestamp'], axis = 1)
        total_user_ratings.columns = ['total_ratings']
        return total_user_ratings
    
    # Function to get the genre ratings
    def UserGenreRatings(self):
        self.genre_ratings = pd.DataFrame()
        for genre in self.genres:
            genre_movies = self.movies.loc[self.movies[genre] == 1]
            avg_genre_votes_per_user = self.ratings[self.ratings['item'].isin(genre_movies.index)].loc[:, ['user', 'rating']].groupby(['user'])['rating'].mean().round(2)
            #print(avg_genre_votes_per_user)
            self.genre_ratings = pd.concat([self.genre_ratings, avg_genre_votes_per_user], axis=1)
        self.genre_ratings = self.genre_ratings.fillna(0)
        self.genre_ratings.columns = self.get_genres()
        return self.genre_ratings
    
    # Function to get Weighted genre ratings for each user
    # weighted by number of genres a user has rated divided by total number of movies rated
    def w_UserGenreRatings(self):
        w1 = pd.DataFrame()
        for genre in self.genres:
            temp = self.UserGenreCounts()[genre].div(self.TotalUserRatings()['total_ratings'])
            w1[genre] = temp
        self.wGR_matrix = self.UserGenreRatings().mul(w1)
        return self.wGR_matrix

    # Function to get the number of ratings per genre per user
    def UserGenreCounts(self):
        genre_counts = pd.DataFrame()
        for genre in self.genres:
            genre_movies = self.movies.loc[self.movies[genre] == 1]
            genre_counts_per_user = self.ratings[self.ratings['item'].isin(genre_movies.index)].loc[:, ['user', 'rating']].groupby(['user'])['rating'].count()
            #print(genre_counts_per_user)
            genre_counts = pd.concat([genre_counts, genre_counts_per_user], axis=1)
        genre_counts = genre_counts.fillna(0)
        genre_counts.columns = self.get_genres()
        return genre_counts
    
    def svd(self, n, dataset='UI'):
        self.UserItem()
        self.UI_SVD =  TruncatedSVD(n_components = n, algorithm = 'arpack')
        self.UI = pd.DataFrame(self.UI_SVD.fit_transform(self.UI_matrix))
        self.UI.index += 1
        return self.UI
    
# CLASS TO CLUSTER AND EVALUATE DATA
class cluster:

    # constructor taking in dataset (UI matrix), number of maximum clusters
    def __init__(self, UI, threshold):
        # assign input rating matrix
        self.UI = UI
        #assign PCA threshold
        self.threshold = threshold
        # Enable logging
        self._logger = logging.getLogger(__name__)
        
    def __str__(self):
        return 'Data Object'
    
    # perform dimensionality reduction to n latent features using SVD
    #def svd(self, n):
    #    SVD =  TruncatedSVD(n_components = n, algorithm = 'arpack')
    #    self.data = pd.DataFrame(SVD.fit_transform(self.UI))
    #    self.data.index += 1
    #    return None
    
    def svd(self):
        pca = PCA(n_components = self.threshold, svd_solver='auto')
        self.data = pd.DataFrame(pca.fit_transform(self.UI))
        self.data.index += 1
        return None
    
    # takes in a list of user_ids, item_ids, and ratings
    def add_interactions(self, user_id, item_id, rating):
        
        assert (
            len(user_id) == len(item_id) == len(rating)
        ), "Input lists are not of the same length"
        
        for i in range(len(user_id)):
            # Check that requirements are met
            assert (user_id[i] in self.UI.index) == True, "User ID does not exist!"
            assert (item_id[i] in self.UI.columns) == True, "Movie ID does not exist!"
            assert (self.UI.loc[user_id[i],item_id[i]] == 0), "User-Item interaction already exists!"
            assert rating[i] > 0, "Rating needs to be [1,5]!"
            assert rating[i] <= 5, "Rating needs to be [1,5]!"
        
            self.UI.loc[user_id[i],item_id[i]] = rating[i]
            
        return None
    
    # return the current user-item interaction matrix
    def get_interactions(self):
        return self.UI
        
    # Function adds user and returns the ID of the new user
    def add_user_auto(self):
        # create new row for new user (ID = len(self.UI)+1), initialise with 0
        self.UI.loc[len(self.UI)+1] = np.zeros(len(self.UI.columns))
        return len(self.UI)
        
    # Function adds movie and returns the ID of the new movie
    def add_item_auto(self):
        # create new column for new movie (ID = len(self.UI.columns)+1), initialise with 0
        self.UI[len(self.UI.columns)+1] = 0
        return len(self.UI.columns)
    
    '''
    # I DON'T THINK WE SHOULD ADD USERS OR MOVIES BY ID AS IT CAN CAUSE A LOT OF CONFUSION DOWN THE LINE.
    # IT MIGHT BE BETTER TO CREATE NEW USERS AND MOVIES IN ORDER AND RETURNING THE ID OF THE NEW ITEMS
    '''
    
    # Function adds user and returns the ID of the new user
    def add_user(self, user_id):
        # Check if user_id to be added already exists
        try:
            assert (user_id in self.UI.index) == False, "User ID already exists!"
        except AssertionError as e:
            print(e)
            exit(1)
            
        # create new row for new user (ID = len(self.UI)+1), initialise with 0
        self.UI.loc[user_id] = np.zeros(len(self.UI.columns))
        return user_id
        
    # Function adds movie and returns the ID of the new movie
    def add_item(self, item_id):
    # Check if item_id to be added already exists
        try:
            assert (item_id in self.UI.columns) == False, "Item ID already exists!"
        except AssertionError as e:
            print(e)
            exit(1)
            
        # create new column for new movie (ID = len(self.UI.columns)+1), initialise with 0
        self.UI[item_id] = 0
        return item_id
            
    # perform kmeans clustering for n clusters on data and return a dataframe with user and cluster number 
    def kmeans(self, n, df='pred', svd = True):
        
        if n is None:
            self._logger.warning('Number of clusters not provided')
            return None
        
        if svd == True:
            # update SVD for dimensionality reducation
            self.svd()
            data = self.data
        else:
            data = self.UI
            
        km = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=0)
        self.km_pred = km.fit_predict(data)
        self.km_pred = pd.DataFrame(self.km_pred, columns = ['cluster'])
        self.km_pred.index += 1 # adjust index to match user
        
        # Return new dataframe with clusters
        if df == 'pred':
            return self.km_pred
        elif df == 'all':
            return [data, self.km_pred]
        else:
            self._logger.error("Invalid input. Enter 'all' or 'pred'.")
            return None
    
    # print graphs to evaluate kmeans clustering from 2 to n clusters using kmeans score, silhouette score and davies-bouldin score
    def kmeans_eval(self, n):
        
        if n is None:
            self._logger.warning('Number of maximum clusters not provided')
            return None
        
        # update SVD for dimensionality reducation
        self.svd()
        
        # variable scope limited to function
        km_scores= []
        km_silhouette = []
        db_score = []
        
        # calculate scores 
        for i in range(2,n+1):
            km = KMeans(n_clusters=i, random_state=0, max_iter=300).fit(self.data)
            km_pred = km.predict(self.data)

            #KM Score
            km_scores.append(-kmeans.score(self.data))

            #Silhouette Score
            km_silhouette.append(metrics.silhouette_score(self.data, km_pred))

            #Davies Bouldin Score
            # the average similarity measure of each cluster with its most similar cluster, 
            # where similarity is the ratio of within-cluster distances to between-cluster distances. 
            # Thus, clusters which are farther apart and less dispersed will result in a better score.
            db_score.append(metrics.davies_bouldin_score(self.data, km_pred))

        # plot graphs of evaluation metrics
        # ELBOW METHOD (optimal cluster at elbow in curve)
        plt.figure(figsize=(14,21))
        plt.subplot(3,1,1)
        plt.title("The elbow method for determining number of clusters",fontsize=16)
        plt.scatter(x=[i for i in range(2,n+1)],y=km_scores,s=150,edgecolor='k')
        plt.grid(True)
        plt.xlabel("Number of clusters",fontsize=14)
        plt.ylabel("K-means Score",fontsize=15)
        plt.xticks([i for i in range(2,n+1)],fontsize=14)
        plt.yticks(fontsize=15)
        
        # SILHOUETTE SCORE (silhouette score varies from [-1,1] with 1 meaning clearly defined clusters)
        plt.subplot(3,1,2)
        plt.title("The silhouette coefficient method for determining number of clusters (1 is ideal)",fontsize=16)
        plt.scatter(x=[i for i in range(2,n+1)],y=km_silhouette,s=150,edgecolor='k')
        plt.grid(True)
        plt.xlabel("Number of clusters",fontsize=14)
        plt.ylabel("Average Silhouette Score",fontsize=15)
        plt.ylim(-1,1)
        plt.xticks([i for i in range(2,n+1)],fontsize=14)
        plt.yticks(fontsize=15)
       
        # DAVIES-BOULDIN SCORE (lower score is better and means more disctinct clusters)
        plt.subplot(3,1,3)
        plt.title("The davies-bouldin coefficient method for determining number of clusters (0 is ideal)",fontsize=16)
        plt.scatter(x=[i for i in range(2,n+1)],y=db_score,s=150,edgecolor='k')
        plt.grid(True)
        plt.xlabel("Number of clusters")
        plt.ylabel("Davies-Bouldin Score")
        plt.ylim(bottom = 0)
        plt.xticks([i for i in range(2,n+1)],fontsize=14)
        plt.yticks(fontsize=15)
        
        plt.show()

    # perform GaussianMixture clustering for n clusters on data and return a dataframe with user and cluster number
    def gmm(self, n, covariance_type='full', df='pred', svd = True):
        # n = number of clusters
        # covariance_type is 'full', 'diag', 'tied' or 'spherical'
        # df is 'pred' for cluster predictions, 'proba' for cluster probabilities, and 'full' for input data combined with probabilities
        if n is None:
            self._logger.warning('Number of maximum clusters not provided')
            return None
        
        if covariance_type is None:
            self._logger.warning('Covariance Type for Gaussian Mixture Model not provided. Default is "full".')
            return None
        
        if df is None:
            self._logger.warning('Return df format not provided. Default is "pred".')
            return None
        
        if svd == True:
            # update SVD for dimensionality reducation
            self.svd()
            data = self.data
        else:
            data = self.UI
        
        gmm = GaussianMixture(n_components=n, n_init=10, covariance_type=covariance_type, tol=1e-3, max_iter=500)
        self.gmm_pred = gmm.fit_predict(data)
        self.gmm_pred = pd.DataFrame(self.gmm_pred, columns = ['cluster'])
        
        # Return new dataframe with clusters, and probability of belonging to a cluster 
        if df == 'pred':
            self.gmm_pred.index += 1
            return self.gmm_pred
        elif df == 'proba':
            cols = ['proba_C'+str(int) for int in range(n)]
            proba = self.gmm_pred.join(pd.DataFrame(gmm.predict_proba(data), columns = cols))
            proba.index += 1 # adjust index to match user
            return proba
        elif df == 'all':
            cols = ['proba_C'+str(int) for int in range(n)]
            proba = self.gmm_pred.join(pd.DataFrame(gmm.predict_proba(data), columns = cols))
            proba.index += 1 # adjust index to match user
            return [data, proba]
        else:
            self._logger.error("Invalid input. Enter 'all', 'pred' or 'proba'.")
            return None
    
    # print graphs to evaluate kmeans clustering from 2 to n clusters using 
    def gmm_eval(self, n, covariance_type="full"):
        
        if n is None:
            self._logger.error('Number of maximum clusters not provided')
            return None
        
        if covariance_type is None:
            self._logger.warning('Covariance Type for Gaussian Mixture Model not provided. Default is "full"')
            return None
        
        # update SVD for dimensionality reducation
        self.svd()
        
        # variable scope limited to function
        gmm_aic = []
        gmm_bic = []
        gmm_scores = [] 
        
        # calculate scores 
        for i in range(2,n+1):
            gmm = GaussianMixture(n_components=i,n_init=10, covariance_type = covariance_type, tol=1e-3,max_iter=500).fit(self.data)
            
            # Akaike Information Criterion
            gmm_aic.append(gmm.aic(self.data))
            
            # Bayesian Information Criterion
            gmm_bic.append(gmm.bic(self.data))
            
            gmm_scores.append(gmm.score(self.data))
            
        # Plot the scores 
        plt.figure(figsize=(14,21))
        plt.subplot(3,1,1)
        #plt.title("The Gaussian Mixture model AIC for determining number of clusters, CT = "+covariance_type,fontsize=16)
        plt.scatter(x=[i for i in range(2,n+1)],y=np.log(gmm_aic),s=150,edgecolor='k')
        plt.grid(True)
        plt.xlabel("Number of clusters",fontsize=14)
        plt.ylabel("Log of Gaussian mixture AIC score",fontsize=15)
        plt.xticks([i for i in range(2,n+1)],fontsize=14)
        plt.yticks(fontsize=15)

        plt.subplot(3,1,2)
        #plt.title("The Gaussian Mixture model BIC for determining number of clusters, CT = "+covariance_type,fontsize=16)
        plt.scatter(x=[i for i in range(2,n+1)],y=np.log(gmm_bic),s=150,edgecolor='k')
        plt.grid(True)
        plt.xlabel("Number of clusters",fontsize=14)
        plt.ylabel("Log of Gaussian mixture BIC score",fontsize=15)
        plt.xticks([i for i in range(2,n+1)],fontsize=14)
        plt.yticks(fontsize=15)
   
        plt.subplot(3,1,3)
        #plt.title("The Gaussian Mixture model scores for determining number of clusters, CT = "+covariance_type,fontsize=16)
        plt.scatter(x=[i for i in range(2,n+1)],y=gmm_scores,s=150,edgecolor='k')
        plt.grid(True)
        plt.xlabel("Number of clusters",fontsize=14)
        plt.ylabel("Gaussian mixture score",fontsize=15)
        plt.xticks([i for i in range(2,n+1)],fontsize=14)
        plt.yticks(fontsize=15)
        plt.show()
        
        return None

    def plot_scatter(self, show_cluster, model):
        
        # logger warning if no clusters to plot/colour  
        if show_cluster:
            if model == 'gmm':
                if self.gmm_pred is None:
                    self._logger.error("Gaussian Mixture Model not trained. Use data.gmm(n, covariance_type, df) to train before plotting")
                    return None
                clusters = self.gmm_pred
            elif model == 'kmeans':
                if self.km_pred is None:
                    self._logger.error("K-Means Model not trained. Use data.kmeans(n) to train before plotting")
                    return None
                clusters = self.km_pred
            marker = {'size': 3,'opacity': 0.8,'color':clusters['cluster'],'colorscale':'Viridis'}
        else:
            marker = {'size': 3,'opacity': 0.8,'colorscale':'Viridis'}
        
        # check input dataset to plot
        if len(self.data.columns) >= 3:
            if len(self.data.columns) > 3:
                self._logger.warning("Input dataset contains more than 3 features. 3D scatter plot will only plot first 3 features.")            
            
            # plot 3D scatter plot
            # Configure Plotly to be rendered inline in the notebook.
            py.init_notebook_mode()
            # Configure the trace.
            trace = go.Scatter3d(
                x=self.data[0],  # <-- Put your data instead
                y=self.data[1],  # <-- Put your data instead
                z=self.data[2],  # <-- Put your data instead
                mode='markers',
                marker=marker
            )
            # Configure the layout.
            layout = go.Layout(
                margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
            )
            data = [trace]
            plot_figure = go.Figure(data=data, layout=layout)
            # Render the plot.
            py.iplot(plot_figure)
            return None

        elif len(self.data.columns) == 2:
            self._logger.warning("Input dataset contains only 2 features. 2D scatter plot will be created.")
            
            # plot 2D scatter plot
            fig = go.Figure(data=go.Scatter(
                x=self.data[0], 
                y=self.data[1], 
                mode='markers', 
                marker=marker))
            fig.show()
            return None
        else:
            self._logger.error("Input dataset contains less than 2 features. Insufficient data to plot.")
            return None
        
#CLASS ANALYSIS TO PROCESS POST-SIMULATION DATA
class analysis:
    
    # constructor taking in dataset (UI matrix), number of maximum clusters
    def __init__(self, probas):
        # assign list of proba dataframes
        self.probas = probas
        # Enable logging
        self._logger = logging.getLogger(__name__)
        self.clusters = self.probas[0]['cluster'].unique().tolist()
        self.cluster_pop = pd.DataFrame()
        
    def __str__(self):
        return 'Analysis Object'
    
    def rename_cluster(self):
        # l and r are indexes of extreme left and extreme right users in synthetic dataset
        # for each iteration i
        for i in range(len(self.probas)):
            # identify cluster names
            groupA = int(self.probas[i][0:99].mode().cluster[0])
            #print(groupA)
            groupB = int(self.probas[i][900:999].mode().cluster[0])
            #print(groupB)
            #groupA = self.probas[i].loc[left_id,'cluster']
            #groupB = self.probas[i].loc[right_id,'cluster']
            
            if len(self.probas[i].cluster.unique()) == 3:
                if groupA == groupB:
                    self._logger.warning("Left and Right Users are in the same cluster. They are both in cluster '1'. Cluster 0 and -1 are both random neutrals now")
                    #groupA = self.probas[i].loc[left_id,'cluster']

                    if (3-groupA) == 3:
                        #groupA is 0
                        groupB = 1
                        groupC = 2
                    else:
                        #groupA is 1 or 2
                        groupB = 3-groupA 
                        groupC = 0
                else:
                    groupC = 3-(groupA+groupB)
                
                #check if it is just predictions or predictions and probabilities 
                if len(self.probas[i].columns) > 2:
                    # rename columns
                    self.probas[i].rename(columns={'proba_C'+str(groupA):1,'proba_C'+str(groupB):-1, 'proba_C'+str(groupC):0},inplace = True)

                # rename clusters
                self.probas[i]['cluster'] = self.probas[i]['cluster'].replace([groupA,groupB,groupC],[1,-1,0])
                self.clusters = [-1,0,1]
                
            elif len(self.probas[i].cluster.unique()) == 2:
                if groupA == groupB:
                    self._logger.warning("Left and Right Users are in the same cluster. They are both in cluster '1'. Cluster 0 is random neutrals now")
                    #groupA = self.probas[i].loc[left_id,'cluster']
                    groupB = 1-groupA
                
                #check if it is just predictions or predictions and probabilities 
                if len(self.probas[i].columns) > 2:
                    # rename columns
                    self.probas[i].rename(columns={'proba_C'+str(groupA):1,'proba_C'+str(groupB):0},inplace = True)

                # rename clusters
                self.probas[i]['cluster'] = self.probas[i]['cluster'].replace([groupA,groupB],[1,0])
                self.clusters = [0,1]
                
        return self.probas
    
    # Function to calculate cluster composition
    def cluster_populations(self):
        if self.probas == None:
            self._logger.error("List of probabilities is empty.")
            return None
        else:
            self.cluster_pop = pd.DataFrame(index=range(1,len(self.probas)+1), columns=(self.clusters + ['total']))
            for t in range(1,len(self.probas)+1):
                for c in self.clusters:
                    self.cluster_pop.loc[t,c] =  len(self.probas[t-1].loc[self.probas[t-1]['cluster']==c])
                self.cluster_pop.loc[t,'total'] = len(self.probas[t-1])
            return self.cluster_pop
            
    def plot_counts(self, show=True, loc=None):
        if self.cluster_pop.empty:
            self.cluster_populations()
        
        plt.clf()
        for i in self.cluster_pop.columns:
            if i == 0:
                color = 'g'
                lbl = "Neutral"
            elif i == 1:
                color = 'b'
                lbl = "Bias 1"
            elif i == -1:
                color = 'r'
                lbl = "Bias 2"
            plt.plot(self.cluster_pop.index,self.cluster_pop[i], color, label = lbl)
       
        plt.xlabel('Iteration')
        # Set the y axis label of the current axis.
        plt.ylabel('Number of users')
        # Set a title of the current axes.
        #plt.title('Change in Cluster Size')
        # show a legend on the plot
        plt.legend(bbox_to_anchor=(1,0.5),loc='center left')
        if show:
            # Display a figure.
            plt.show(block=True)
        else:
            #save plt to loc
            plt.savefig(loc, bbox_inches='tight')
        
    def plot_percent(self, show=True, loc=None):
        if self.cluster_pop.empty:
            self.cluster_populations()
        
        plt.clf()
        for i in self.cluster_pop.columns[:-1]:
            if i == 0:
                color = 'g'
                lbl = "Neutral Community"
            elif i == 1:
                color = 'b'
                lbl = "Bias Community 1"
            elif i == -1:
                color = 'r'
                lbl = "Bias Community 2"
            plt.plot(self.cluster_pop.index,(self.cluster_pop[i]/self.cluster_pop['total']), color, label = lbl)

        plt.ylim(0,1)
        plt.xlabel('Iteration')
        # Set the y axis label of the current axis.
        plt.ylabel('Fraction of Users')
        # Set a title of the current axes.
        #plt.title('Change in Cluster Size')
        # show a legend on the plot
        plt.legend(bbox_to_anchor=(0.5,-0.2),loc='upper center')
        if show:
            # Display a figure.
            plt.show(block=True)
        else:
            #save plt to loc
            plt.savefig(loc, bbox_inches='tight')
       
    '''
    # Function to calculate adjacency matrix of weighted graph of users. Default similarity algorithm is Jaccard
    def adj_matrix(self, sim = "cosin"):
        # calculate similarity matrix 
        if sim == "cosin":
            print("cosin")
        else:
            self._logger.error("Invalid input for sim. Enter 'cosin' or 'euclidean'.")
            return None
    '''
            
class post_process():
        
    # constructor taking in dataset (UI matrix), number of maximum clusters
    def __init__(self, latent_list, results_list, UI):
        assert len(latent_list) == len(results_list), "Length of latents and results does not match"
        assert len(results_list[len(results_list)-1]) == len(UI[len(UI)-1]), "Number of users in final cluster results does not match the number of users in the final user-item interaction matrix"
        
        # assign input list of DFs
        self.latents = latent_list
        self.results = results_list
        self.UI = UI
        
        # Enable logging
        self._logger = logging.getLogger(__name__)

        # Create Analysis Object for Clustering Probabilities
        self.analysis_obj = analysis(self.results)
        
    def __str__(self):
        return 'Post Processing Object'
    
    def rename_cluster(self,l,r):
        # l and r are indexes of extreme left and extreme right users in synthetic dataset
        return self.analysis_obj.rename_cluster()
    
    # Function to calculate cluster composition
    def cluster_populations(self):
        return self.analysis_obj.cluster_populations()
            
    def plot_counts(self, show=True, loc=None):
        self.analysis_obj.plot_counts(show=show, loc=loc)
        
    def plot_percent(self, show=True, loc=None):
        self.analysis_obj.plot_percent(show=show, loc=loc) 
        
    def examine(self, i, algo, camera_eye=dict(x=1.345,y=1.324,z=1.08), camera_center=dict(x=0.095,y=0.074,z=-0.16)):
    # n is the number of clusters
    
        # ensure number of 
        if i >= len(self.latents):
            self._logger.error("Argument 'i' is out of bounds. It needs to be between 0 and %d" % (len(self.latents)-1))
            return None
        
        if not((algo == 'gmm') or (algo == 'kmeans')):
            self._logger.error("Argument 'algo' needs to be either 'gmm' or 'kmeans'.")
            return None
        
        n = len(self.results[0].cluster.unique())
        
        if (algo == 'gmm'):
            gmm = GaussianMixture(n_components=n, n_init=10, covariance_type='full', tol=1e-3, max_iter=500)
            pred = gmm.fit_predict(self.latents[i])
            pred = pd.DataFrame(pred, columns = ['cluster'])
            pred.index += 1 # adjust index to match user
            
        elif (algo == 'kmeans'):
            km = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=0)
            pred = km.fit_predict(self.latents[i])
            pred = pd.DataFrame(pred, columns = ['cluster'])
            pred.index += 1 # adjust index to match user
            
        figure = self.plot_scatter(i, pred, show_cluster = True, camera_eye=camera_eye, camera_center=camera_center)
        return(figure)
    
    def save_svg(self, i, algo, loc, camera_eye=dict(x=1.345,y=1.324,z=1.08), camera_center=dict(x=0.095,y=0.074,z=-0.16)):
        fig = self.examine(i, algo, camera_eye=camera_eye, camera_center=camera_center)
        fig.write_image(loc+".svg")
        
    def scene_camera(self, i, pred, show_cluster):
        # Assign colours based on clusters
        if show_cluster:
            marker0 = {'size': 3,'opacity': 0.8,'color':'rgb(0, 157, 0)'}
            marker1 = {'size': 3,'opacity': 0.8,'color':'rgb(0, 0, 157)'}
            marker2 = {'size': 3,'opacity': 0.8,'color':'rgb(157, 0, 0)'}
        else:
            marker0 = {'size': 3,'opacity': 0.8,'color':'rgb(0, 0, 0)'}
            marker1 = {'size': 3,'opacity': 0.8,'color':'rgb(0, 0, 0)'}
            marker2 = {'size': 3,'opacity': 0.8,'color':'rgb(0, 0, 0)'}
        
        # check input dataset to plot
        if len(self.latents[i].columns) >= 3:
            # Configure the 0 trace.
            neutral = go.Scatter3d(
                name = 'neutral',
                x=self.latents[i][self.results[i].cluster == 0][0],  # <-- Put your data instead
                y=self.latents[i][self.results[i].cluster == 0][1],  # <-- Put your data instead
                z=self.latents[i][self.results[i].cluster == 0][2],  # <-- Put your data instead
                mode='markers',
                marker=marker0
            )
            # Configure the 1 trace.
            bias_1 = go.Scatter3d(
                name = 'bias_1',
                x=self.latents[i][self.results[i].cluster == 1][0],  # <-- Put your data instead
                y=self.latents[i][self.results[i].cluster == 1][1],  # <-- Put your data instead
                z=self.latents[i][self.results[i].cluster == 1][2],  # <-- Put your data instead
                mode='markers',
                marker=marker1
            )
            data = [neutral,bias_1]
            
            if (len(self.results[i].cluster.unique())==3):
                # Configure the -1 trace.
                bias_2 = go.Scatter3d(
                    name = 'bias_2',
                    x=self.latents[i][self.results[i].cluster == -1][0],  # <-- Put your data instead
                    y=self.latents[i][self.results[i].cluster == -1][1],  # <-- Put your data instead
                    z=self.latents[i][self.results[i].cluster == -1][2],  # <-- Put your data instead
                    mode='markers',
                     marker=marker2
                )
                data.append(bias_2)
        else:
            # Latents contains 2 or less latent features to reach 30% explained variance. 
            # calculate latents from UI
            temp = cluster(self.UI[i],3)
            temp.svd()
            temp_lats = temp.data
            
            # create traces
            # Configure the 0 trace.
            neutral = go.Scatter3d(
                name = 'neutral',
                x=temp_lats[i][self.results[i].cluster == 0][0],  # <-- Put your data instead
                y=temp_lats[i][self.results[i].cluster == 0][1],  # <-- Put your data instead
                z=temp_lats[i][self.results[i].cluster == 0][2],  # <-- Put your data instead
                mode='markers',
                marker=marker0
            )
            # Configure the 1 trace.
            bias_1 = go.Scatter3d(
                name = 'bias_1',
                x=temp_lats[i][self.results[i].cluster == 1][0],  # <-- Put your data instead
                y=temp_lats[i][self.results[i].cluster == 1][1],  # <-- Put your data instead
                z=temp_lats[i][self.results[i].cluster == 1][2],  # <-- Put your data instead
                mode='markers',
                marker=marker1
            )
            data = [neutral,bias_1]
            
            if (len(self.results[i].cluster.unique())==3):
                # Configure the -1 trace.
                bias_2 = go.Scatter3d(
                    name = 'bias_2',
                    x=temp_lats[i][self.results[i].cluster == -1][0],  # <-- Put your data instead
                    y=temp_lats[i][self.results[i].cluster == -1][1],  # <-- Put your data instead
                    z=temp_lats[i][self.results[i].cluster == -1][2],  # <-- Put your data instead
                    mode='markers',
                    marker=marker2
                )
                data.append(bias_2)
                
        if len(self.latents[i].columns) > 3:
            self._logger.warning("30% variance explained by MORE than 3 features. 3D scatter plot will only plot first 3 features.")            
            
        elif len(self.latents[i].columns) < 3:
            self._logger.warning("30% variance explained by LESS than 3 features. 3D scatter plot will plot first 3 features.")            
            
        # plot 3D scatter plot
        # Configure Plotly to be rendered inline in the notebook.
        py.init_notebook_mode()

        # Configure the layout.
        layout = go.Layout(margin={'l': 10, 'r': 10, 'b': 10, 't': 10})
        #plot_figure=go.FigureWidget(data=data, layout=layout)
        
        plot_figure = go.FigureWidget(data=data, layout=layout) 
        return plot_figure
            
    def plot_scatter(self, i, pred, show_cluster, camera_eye=dict(x=1.345,y=1.324,z=1.08), camera_center=dict(x=0.095,y=0.074,z=-0.16)):
        # Assign colours based on clusters
        if show_cluster:
            marker0 = {'size': 3,'opacity': 0.8,'color':'rgb(0, 157, 0)'}
            marker1 = {'size': 3,'opacity': 0.8,'color':'rgb(0, 0, 157)'}
            marker2 = {'size': 3,'opacity': 0.8,'color':'rgb(157, 0, 0)'}
        else:
            marker0 = {'size': 3,'opacity': 0.8,'color':'rgb(0, 0, 0)'}
            marker1 = {'size': 3,'opacity': 0.8,'color':'rgb(0, 0, 0)'}
            marker2 = {'size': 3,'opacity': 0.8,'color':'rgb(0, 0, 0)'}
        
        # check input dataset to plot
        if len(self.latents[i].columns) >= 3:
            # Configure the 0 trace.
            neutral = go.Scatter3d(
                name = 'Neutral Community',
                x=self.latents[i][self.results[i].cluster == 0][0],  # <-- Put your data instead
                y=self.latents[i][self.results[i].cluster == 0][1],  # <-- Put your data instead
                z=self.latents[i][self.results[i].cluster == 0][2],  # <-- Put your data instead
                mode='markers',
                marker=marker0
            )
            # Configure the 1 trace.
            bias_1 = go.Scatter3d(
                name = 'Bias Community 1',
                x=self.latents[i][self.results[i].cluster == 1][0],  # <-- Put your data instead
                y=self.latents[i][self.results[i].cluster == 1][1],  # <-- Put your data instead
                z=self.latents[i][self.results[i].cluster == 1][2],  # <-- Put your data instead
                mode='markers',
                marker=marker1
            )
            data = [neutral,bias_1]
            
            if (len(self.results[i].cluster.unique())==3):
                # Configure the -1 trace.
                bias_2 = go.Scatter3d(
                    name = 'Bias Community 2',
                    x=self.latents[i][self.results[i].cluster == -1][0],  # <-- Put your data instead
                    y=self.latents[i][self.results[i].cluster == -1][1],  # <-- Put your data instead
                    z=self.latents[i][self.results[i].cluster == -1][2],  # <-- Put your data instead
                    mode='markers',
                     marker=marker2
                )
                data.append(bias_2)
        else:
            # Latents contains 2 or less latent features to reach 30% explained variance. 
            # calculate latents from UI
            temp = cluster(self.UI[i],3)
            temp.svd()
            temp_lats = temp.data
            
            # create traces
            # Configure the 0 trace.
            neutral = go.Scatter3d(
                name = 'Neutral Community',
                x=temp_lats[i][self.results[i].cluster == 0][0],  # <-- Put your data instead
                y=temp_lats[i][self.results[i].cluster == 0][1],  # <-- Put your data instead
                z=temp_lats[i][self.results[i].cluster == 0][2],  # <-- Put your data instead
                mode='markers',
                marker=marker0
            )
            # Configure the 1 trace.
            bias_1 = go.Scatter3d(
                name = 'Bias Community 1',
                x=temp_lats[i][self.results[i].cluster == 1][0],  # <-- Put your data instead
                y=temp_lats[i][self.results[i].cluster == 1][1],  # <-- Put your data instead
                z=temp_lats[i][self.results[i].cluster == 1][2],  # <-- Put your data instead
                mode='markers',
                marker=marker1
            )
            data = [neutral,bias_1]
            
            if (len(self.results[i].cluster.unique())==3):
                # Configure the -1 trace.
                bias_2 = go.Scatter3d(
                    name = 'Bias Community 2',
                    x=temp_lats[i][self.results[i].cluster == -1][0],  # <-- Put your data instead
                    y=temp_lats[i][self.results[i].cluster == -1][1],  # <-- Put your data instead
                    z=temp_lats[i][self.results[i].cluster == -1][2],  # <-- Put your data instead
                    mode='markers',
                    marker=marker2
                )
                data.append(bias_2)
                
        if len(self.latents[i].columns) > 3:
            self._logger.warning("30% variance explained by MORE than 3 features. 3D scatter plot will only plot first 3 features.")            
            
        elif len(self.latents[i].columns) < 3:
            self._logger.warning("30% variance explained by LESS than 3 features. 3D scatter plot will plot first 3 features.")            
            
        # plot 3D scatter plot
        # Configure Plotly to be rendered inline in the notebook.
        py.init_notebook_mode()

        # Configure the layout.
        layout = go.Layout(margin={'l': 10, 'r': 10, 'b': 10, 't': 10})
        #plot_figure=go.FigureWidget(data=data, layout=layout)
        
        plot_figure = go.Figure(data=data, layout=layout)
        plot_figure.update_layout(scene = dict(
                    xaxis_title='Latent Feature 1',
                    yaxis_title='Latent Feature 2',
                    zaxis_title='Latent Feature 3',
                    xaxis = dict(nticks=5, range=[-25,46], gridcolor="black"),
                    yaxis = dict(nticks=5, range=[-25,38], gridcolor="black"),
                    zaxis = dict(nticks=5, range=[-23,34], gridcolor="black")),
                                 height=700,
                                 scene_aspectmode='cube',
                                 font_size=15,
                                 showlegend=None,
                                 legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5,
                    itemwidth=50),
                                 legend_font_size=7)
        plot_figure.update_scenes(camera_eye=camera_eye,
                                 camera_center=camera_center)
        
        # Render the plot.
        py.iplot(plot_figure)
        
        return plot_figure


'''
APPENDIX OF PREVIOUS FUNCTIONS:

From MovieLens Class:
    
    # Function to get the genre ratings
    def UserGenreRatings(self):
        self.genre_ratings = pd.DataFrame()
        for genre in self.genres:
            genre_movies = self.movies[self.movies['genres'].str.contains(genre)]
            avg_genre_votes_per_user = self.ratings[self.ratings['item'].isin(genre_movies['item'])].loc[:, ['user', 'rating']].groupby(['user'])['rating'].mean().round(2)
            self.genre_ratings = pd.concat([self.genre_ratings, avg_genre_votes_per_user], axis=1)
        self.genre_ratings = self.genre_ratings.fillna(0)
        self.genre_ratings.columns = self.get_genres()
        return self.genre_ratings
    
    # Function to get Weighted genre ratings for each user
    # weighted by number of genres a user has rated divided by total number of movies rated
    def w_UserGenreRatings(self):
        w1 = pd.DataFrame()
        for genre in self.genres:
            temp = self.UserGenreCounts()[genre].div(self.TotalUserRatings()['total_ratings'])
            w1[genre] = temp
        self.wGR_matrix = dataGR.mul(w1)
        return self.wGR_matrix

    # Function to get the number of ratings per genre per user
    def UserGenreCounts(self):
        self.genre_counts = pd.DataFrame()
        for genre in self.genres:
            genre_movies = self.movies[self.movies['genres'].str.contains(genre) ]
            genre_counts_per_user = self.ratings[self.ratings['item'].isin(genre_movies['item'])].loc[:, ['user', 'rating']].groupby(['user'])['rating'].count()
            self.genre_counts = pd.concat([self.genre_counts, genre_counts_per_user], axis=1).fillna(0)
        self.genre_counts.columns = self.genres
        return self.genre_counts
        
    def SVD(self, dataset):
        if dataset == 'UI':
            self.UserItem()
            self.UI_SVD =  TruncatedSVD(n_components = n)
            self.UI = pd.DataFrame(self.UI_SVD.fit_transform(self.UI_matrix))
            self.UI.index += 1
            return self.UI
        elif dataset == 'GR':
            self.UserGenreRatings()
            self.GR_SVD =  TruncatedSVD(n_components = n)
            self.GR = pd.DataFrame(self.GR_SVD.fit_transform(self.genre_ratings))
            self.GR.index += 1
            return self.GR
        elif dataset == 'wGR':
            self.w_UserGenreRatings()
            self.wGR_SVD =  TruncatedSVD(n_components = n)
            self.wGR = pd.DataFrame(self.wGR_SVD.fit_transform(self.wGR_matrix))
            self.wGR.index += 1
            return self.wGR
'''



