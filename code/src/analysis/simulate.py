import pandas as pd
import numpy as np
import math

import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt

import logging

# THIS CAN BE INSIDE THE CLASS BASED ON CHOSEN ALGORITHM
sys.path.append("/Users/pvs262/Documents/rec-sys-dynamics/code")
from src.analysis.cluster import cluster, analysis
from src.algorithm.cosin import CosinSimilarity 
from src.algorithm.ease import EASE 
from src.algorithm.mf import MatrixFactorization 
import src.Datasets as datasets

# function to convert dataset names to acronyms
def shortform(str):
    short = ""
    for i in range(len(str)):
        if str[i].isdigit() or str[i].isupper():
            short +=str[i]
    return short

class simulate():
    
    def __init__(self, recsys, ds_name):
        
        # Enable logging
        self._logger = logging.getLogger(__name__)
        
        # import dataset based on ds_name
        self.dataset = '/'+ds_name
        self.run_name = recsys+'_'+shortform(ds_name)+'_'
        # assign simulation parameters
        #self.prob_explore = prob_explore
        #self.new_i = n_i
        #self.new_u = n_u
        #self.num_rec = n_r
        #self.simulation_steps = steps
        self.ratings = self.getRatingsData()
        self.Full_Known_Utilities = self.getUtilityData()
        
        if recsys == 'ease':
            # use EASE algorithm
            self.algo = EASE()
        elif recsys == 'cosin':
            # use CosinSimilarity algorithm
            self.algo = CosinSimilarity()
        elif recsys == 'mf':
            # use Matrix Factorisation algorithm
            self.algo = MatrixFactorization()
        else:
            # use popularity based algorithm
            self._logger.error("Invalid input. recsys has to be 'cosin', 'ease' or 'mf'.")
            return None
        
        self.algo.fit(self.ratings)
        
    def __str__(self):
        return 'Simulate Object'
        
    def getRatingsData(self):
        return pd.read_parquet(os.path.dirname(datasets.__file__)+self.dataset+'/ratings.parquet.gzip')

    def getUtilityData(self):
        return pd.read_pickle(os.path.dirname(datasets.__file__)+self.dataset+'/utility_Matrix.pkl')

    def conv_index_to_bins(self, index):
        """Calculate bins to contain the index values.
        The start and end bin boundaries are linearly extrapolated from 
        the two first and last values. The middle bin boundaries are 
        midpoints.

        Example 1: [0, 1] -> [-0.5, 0.5, 1.5]
        Example 2: [0, 1, 4] -> [-0.5, 0.5, 2.5, 5.5]
        Example 3: [4, 1, 0] -> [5.5, 2.5, 0.5, -0.5]"""
        assert index.is_monotonic_increasing or index.is_monotonic_decreasing

        # the beginning and end values are guessed from first and last two
        start = index[0] - (index[1]-index[0])/2
        end = index[-1] + (index[-1]-index[-2])/2

        # the middle values are the midpoints
        middle = pd.DataFrame({'m1': index[:-1], 'p1': index[1:]})
        middle = middle['m1'] + (middle['p1']-middle['m1'])/2

        if isinstance(index, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(middle).union([start,end])
        elif isinstance(index, (pd.Float64Index,pd.RangeIndex,pd.Int64Index)):
            idx = pd.Float64Index(middle).union([start,end])
        else:
            print('Warning: guessing what to do with index type %s' % 
                  type(index))
            idx = pd.Float64Index(middle).union([start,end])

        return idx.sort_values(ascending=index.is_monotonic_increasing)

    def calc_df_mesh(self, df):
        """Calculate the two-dimensional bins to hold the index and 
        column values."""
        return np.meshgrid(conv_index_to_bins(df.index),
                           conv_index_to_bins(df.columns))

    def heatmap(self, df):
        """Plot a heatmap of the dataframe values using the index and 
        columns"""
        X,Y = calc_df_mesh(self, df)
        c = plt.pcolormesh(X, Y, df.values.T, cmap = 'gist_yarg')
        plt.colorbar(c)   

    def get_rank_func(self, recs, func_type = "sigmoid", explore = False):
        x = np.array([r for r in range(1,recs + 1,1)])
        if explore:
            func = x/x
        else:    
            if func_type =="sigmoid":
                a = recs/10
                b = recs*3/4
                func = 1/(1+math.e**((x-b)/a))
            if func_type == "dec_exp":
                func = x**(-0.8)
        return func            

    def env_step(self, P_df, prob_explore,num_rec):
        inter_users=[]
        inter_items =[]
        inter_ratings=[]
        policy_choices = np.random.choice(['Explore', 'Exploit'], len(P_df.index), p = [prob_explore, 1-prob_explore])
        for user in range(1, len(P_df.index)+1):
            policy = policy_choices[user-1]
            if policy == "Explore":
                recs = self.algo.recommend(user, explore=True)
                func_rank = self.get_rank_func(num_rec, explore = True)
            if policy == "Exploit":
                recs = self.algo.recommend(user)
                func_rank = self.get_rank_func(num_rec)
            user_utilities_ordered = P_df.loc[user].reindex(recs.index).to_numpy()[0:num_rec]
            func = func_rank*user_utilities_ordered
            max_index = np.where(func == max(func))
            item_2_rec = int(recs.index[max_index][0])
            given_rating =  self.get_rating(P_df.loc[user], item_2_rec)
            rating_flip = np.random.choice([True, False], p = [0.1,0.9])
            if rating_flip:
                pos_ratings = [1,2,3,4,5]
                pos_ratings.remove(given_rating)
                given_rating = np.random.choice(pos_ratings)
            inter_users.append(user);inter_items.append(item_2_rec); inter_ratings.append(given_rating)

        interactions_by_iteration = [inter_users, inter_items, inter_ratings]

        return  interactions_by_iteration 


    def add_users(self, num_u, simulation_step, new_u):
        if simulation_step!=0:
            for user in [*range(num_u + (simulation_step-1)*new_u + 1, num_u + (simulation_step)*new_u + 1,1)]:
                self.algo.add_user(user)
                self.cluster_obj.add_user(user)

    def add_items(self, num_i, simulation_step, new_i):
        if simulation_step!=0:
            for item in [*range(num_i + (simulation_step-1)*new_i + 1, num_i + (simulation_step)*new_i + 1,1)]:
                self.algo.add_item(item)
                self.cluster_obj.add_item(item)

    def add_interactions(self, list_of_interactions):
        self.algo.add_interactions(list_of_interactions[0], list_of_interactions[1], list_of_interactions[2])
        self.cluster_obj.add_interactions(list_of_interactions[0], list_of_interactions[1], list_of_interactions[2])

    def get_rating(self, P_df_u_values, item):
        return round(1 + ((P_df_u_values[item]-P_df_u_values.min())*(5-1))/(P_df_u_values.max()-P_df_u_values.min()))

    def get_probas(self, cluster_obj, n_clusters):
        return cluster_obj.gmm(n=n_clusters,covariance_type="full",df='all',svd=True)

    def run_dynamics(self, n_i, n_u, n_r, steps, prob_explore = 0.2, svd_threshold=0.3, n_clusters=3):
        # assign simulation parameters
        new_i = n_i
        new_u = n_u
        num_rec = n_r
        simulation_steps = steps

        # create user-item interaction matrix from ratings
        ratings_df = pd.DataFrame(self.algo.rating_matrix_.todense(), columns = range(1,self.algo.rating_matrix_.shape[1]+1),index = range(1,self.algo.rating_matrix_.shape[0]+1))
        
        # create cluster object with 
        self.cluster_obj = cluster(ratings_df,svd_threshold)
        
        num_u = len(ratings_df)
        num_i = len(ratings_df.columns)

        # Intializing the full utilities matrix
        # StartTime = datetime.now()
        # print("Initizing Full Utility Matrix.......")
        # Full_Known_Utilities = initialize_known_utility(num_u, num_i, simulation_steps, new_i, new_u)
        # print("Initization of Full Utility Matrix Complete")
        # print("Time taken: ", datetime.now() - StartTime)


        # Storing max utilities by user for thresholding
        # maximum_util_by_user = P_df.max(axis=1)
        results = []
        latents = []

        interactions = [[],[],[]]
        
        # iterate over 
        for t in range(simulation_steps):
            StartTime = datetime.now()
            print("Starting iteration number: ", t+1)

            # interactions is arranged as a 3xn array of arrays, where n is the number of interactions. 
            # In intercations, dimension 1 is the users that interact, dimension 2 is the items that they interact with, dimension 3 is the ratings given.
            self.add_users(num_u, t, new_u)
            self.add_items(num_i, t, new_i)
            self.add_interactions(interactions)
            self.algo.update()

            P_df_t = self.Full_Known_Utilities.iloc[0:(num_u+(t*new_u)),0:(num_i+(t*new_i))]

            # probability currently kept constant
            # change prob explore to a decreasing function to have a reudcing explore prob
            # prob_explore = prob_explore

            interactions = self.env_step(P_df_t, prob_explore,num_rec)

            # if np.isnan(sum(interactions[2])):
            #     print("There are nan being added")

            results.append(self.get_probas(self.cluster_obj, n_clusters)[1])
            latents.append(self.get_probas(self.cluster_obj, n_clusters)[0])

            print("Completed iteration number: ", t+1)
            print("Time taken: ", datetime.now() - StartTime)

        updated_ratings_df = pd.DataFrame(self.algo.rating_matrix_.toarray(), columns=np.arange(1,len(self.algo.rating_matrix_.toarray()[0])+1), index=np.arange(1,len(self.algo.rating_matrix_.toarray())+1))
        
        # save pickles of latents, results + final_UIs
        self.run_name = '../simulation_runs/'+self.run_name+str(n_i)+'_'+str(n_u)+'_'+str(n_r)+'_'+str(steps)
        os.makedirs(self.run_name)
        # save updated ratings
        updated_ratings_df.to_pickle(self.run_name+'/final_UI.pkl.gzip', compression = 'gzip')

        for i in range(len(latents)):
            latents[i].to_pickle(self.run_name+'/L'+str(i)+'pkl.gzip', compression = 'gzip')
            results[i].to_pickle(self.run_name+'/R'+str(i)+'pkl.gzip', compression = 'gzip')
            
        # save the plot_counts() and plot_percent pngs
        #run = analysis(results)
        #run.plot_counts(show=False, loc=directory+'/counts.png')
        #run.plot_percent(show=False, loc=directory+'/percent.png')
        
        return [latents, results, updated_ratings_df]


