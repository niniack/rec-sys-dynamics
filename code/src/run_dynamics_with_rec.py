import pandas as pd
import numpy as np
import math

import os
from datetime import datetime
from algorithm.ease import EASE
import Datasets.movielens_recreate as dataset
import matplotlib.pyplot as plt
from analysis.cluster import cluster


def getRatingsData():
    return pd.read_parquet(os.path.dirname(dataset.__file__)+'/ratings.parquet.gzip')

def getUtilityData():
    return pd.read_pickle(os.path.dirname(dataset.__file__)+'/utility_Matrix.pkl')


ratings = getRatingsData()
Full_Known_Utilities = getUtilityData()

algo = EASE()

algo.fit(ratings)

def conv_index_to_bins(index):
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

def calc_df_mesh(df):
    """Calculate the two-dimensional bins to hold the index and 
    column values."""
    return np.meshgrid(conv_index_to_bins(df.index),
                       conv_index_to_bins(df.columns))

def heatmap(df):
    """Plot a heatmap of the dataframe values using the index and 
    columns"""
    X,Y = calc_df_mesh(df)
    c = plt.pcolormesh(X, Y, df.values.T, cmap = 'gist_yarg')
    plt.colorbar(c)   
    
def get_rank_func(recs, func_type = "sigmoid", explore = False):
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

def env_step(P_df, prob_explore,num_rec):
    inter_users=[]
    inter_items =[]
    inter_ratings=[]
    policy_choices = np.random.choice(['Explore', 'Exploit'], len(P_df.index), p = [prob_explore, 1-prob_explore])
    for user in range(1, len(P_df.index)+1):
        policy = policy_choices[user-1]
        if policy == "Explore":
            recs = algo.recommend(user, explore=True)
            func_rank = get_rank_func(num_rec, explore = True)
        if policy == "Exploit":
            recs = algo.recommend(user)
            func_rank = get_rank_func(num_rec)
        user_utilities_ordered = P_df.loc[user].reindex(recs.index).to_numpy()[0:num_rec]
        func = func_rank*user_utilities_ordered
        max_index = np.where(func == max(func))
        item_2_rec = int(recs.index[max_index][0])
        given_rating =  get_rating(P_df.loc[user], item_2_rec)
        rating_flip = np.random.choice([True, False], p = [0.1,0.9])
        if rating_flip:
            pos_ratings = [1,2,3,4,5]
            pos_ratings.remove(given_rating)
            given_rating = np.random.choice(pos_ratings)
        inter_users.append(user);inter_items.append(item_2_rec); inter_ratings.append(given_rating)
    
    interactions_by_iteration = [inter_users, inter_items, inter_ratings]
        
    return  interactions_by_iteration 


def add_users(num_u, simulation_step, new_u):
    if simulation_step!=0:
        for user in [*range(num_u + (simulation_step-1)*new_u + 1, num_u + (simulation_step)*new_u + 1,1)]:
            algo.add_user(user)
            # cluster_obj.addUserID(user)
            
def add_items(num_i, simulation_step, new_i):
    if simulation_step!=0:
        for item in [*range(num_i + (simulation_step-1)*new_i + 1, num_i + (simulation_step)*new_i + 1,1)]:
            algo.add_item(item)
            # cluster_obj.addItemID(item)
        
def add_interactions(list_of_interactions):
    algo.add_interactions(list_of_interactions[0], list_of_interactions[1], list_of_interactions[2])
    # cluster_obj.addRating(list_of_interactions[0], list_of_interactions[1], list_of_interactions[2])
    
def get_rating(P_df_u_values, item):
    return round(1 + ((P_df_u_values[item]-P_df_u_values.min())*(5-1))/(P_df_u_values.max()-P_df_u_values.min()))

def get_probas(cluster_obj):
    return cluster_obj.gmm(n=3,covariance_type="full",df='proba')
    


num_u = 943
# num_u = 5
num_i = 1682
# num_i = 5
num_rec = 30
simulation_steps = 5
new_i = 10
new_u = 10

ratings_df = pd.DataFrame(algo.rating_matrix_.toarray(), index = np.arange(1, num_u+1,1), columns = np.arange(1, num_i+1,1))

cluster_obj = cluster(ratings_df)
cluster_obj.SVD(3)

# # Intializing teh full utilities matrix
# StartTime = datetime.now()
# print("Initizing Full Utility Matrix.......")
# Full_Known_Utilities = initialize_known_utility(num_u, num_i, simulation_steps, new_i, new_u)
# print("Initization of Full Utility Matrix Complete")
# print("Time taken: ", datetime.now() - StartTime)


#storing max utilities by user for thresholding
# maximum_util_by_user = P_df.max(axis=1)
probabilities = []

interactions = [[],[],[]]
for t in range(simulation_steps):
    StartTime = datetime.now()
    print("Starting iteration number: ", t+1)
    
    # interactions is arranged as a 3xn array of arrays, where n is the number of interactions. 
    # In intercations, dimension 1 is the users that interact, dimension 2 is the items that they interact with, dimension 3 is the ratings given.
    add_users(num_u, t, new_i)
    add_items(num_i, t, new_i)
    add_interactions(interactions)
    algo.update()
    
    P_df_t = Full_Known_Utilities.iloc[0:(num_u+(t*new_u)),0:(num_i+(t*new_i))]
    
    # probability currently kept constant
    # change prob explore to a decreasing function to have a reudcing explore prob
    prob_explore = 0.4
    
    
    interactions = env_step(P_df_t, prob_explore,num_rec)
    
    # if np.isnan(sum(interactions[2])):
    #     print("There are nan ratings being added")
    
    
    probabilities.append(get_probas(cluster_obj))
    
    
    print("Completed iteration number: ", t+1)
    print("Time taken: ", datetime.now() - StartTime)
    
   
    
updated_ratings_df = pd.DataFrame(algo.rating_matrix_.toarray(), columns=np.arange(1,len(algo.rating_matrix_.toarray()[0])+1), index=np.arange(1,len(algo.rating_matrix_.toarray())+1))



