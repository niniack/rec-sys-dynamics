import pandas as pd
import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import math

import sys
from pprintpp import pprint as prettyprint
from datetime import datetime

#sys.path.append("..")

from algorithm.cosin import CosinSimilarity

from lenskit.datasets import ML100K

movielens = ML100K("./ml-100k")
ratings = movielens.ratings
#ratings.head()

algo_cosin = CosinSimilarity()

algo_cosin.fit(ratings)


def get_mu(v1, v2):
    mu = np.dot(v1,v2)
    return mu

def get_alpha_beta(mu, sigma=10**(-5)):
    a1 = (mu**2)*(((1-mu)/sigma**2)-(1/mu))
    b1 = a1*((1/mu)- 1)
    return a1, b1

def get_Vui(u, i, p, a):
    B_mu = get_mu(p[u-1],a[i-1])
    alpha, beta = get_alpha_beta(B_mu)
    return float(np.random.beta(alpha,beta))

def get_nui(mu = 0.98):
    alpha, beta = get_alpha_beta(mu)
    return float(np.random.beta(alpha,beta))

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
    
def get_rank_func(recs, func_type = "sigmoid"):
    x = np.array([r for r in range(1,recs + 1,1)])
    if func_type =="sigmoid":
        a = recs/10
        b = recs*3/4
        func = 1/(1+math.e**((x-b)/a))
    if func_type == "dec_exp":
        func = x**(-0.8)
    return func

def initialize_known_utility(num_u, num_i, total_sim_steps, new_i, new_u, alpha_user=1, alpha_item = 100, K_pref_dimensions = 20):
    
    n_i = num_i + (total_sim_steps*new_i)
    n_u = num_u + (total_sim_steps*new_u)
    
    col = np.arange(1, n_i+1, 1)
    empty_data_row = [float(0) for item in range(1, n_i+1)]
    empty_data = np.array([empty_data_row for item in range(1, n_u+2)])
    
    V_df = pd.DataFrame(empty_data, columns = col)
    P_df = pd.DataFrame(empty_data, columns = col)
    V_df = V_df.drop([0], axis = 0)
    P_df = P_df.drop([0], axis = 0)
    
    p = dirichlet.rvs([alpha_user]*K_pref_dimensions, size=n_u)
    a = dirichlet.rvs([alpha_item]*K_pref_dimensions, size=n_i)
    
    for item in range(1,n_i+1):
        for user in range(1, n_u+1):
            Vui = get_Vui(user,item, p, a)
            V_df[item][user] = Vui
            known_proportion = get_nui()
            P_df[item][user] = Vui*known_proportion
            
    return P_df
            



def env_step(P_df, prob_explore,num_rec):
    inter_users=[]
    inter_items =[]
    inter_ratings=[]
    policy_choices = np.random.choice(['Explore', 'Exploit'], len(P_df.index), p = [prob_explore, 1-prob_explore])
    for user in range(1, len(P_df.index)+1):
        policy = policy_choices[user-1]
        recs = algo_cosin.recommend(user)
        if policy == "Explore":
            recs = recs.sort_values(by=["normalized_popularity","predicted_ratings"], ascending=[True,False])[["predicted_ratings", "normalized_popularity"]]
        if policy == "Exploit":
            recs = recs.sort_values(by=["predicted_ratings","normalized_popularity"], ascending=False)[["predicted_ratings", "normalized_popularity"]]
        user_utilities_ordered = P_df.loc[user].reindex(recs.index).to_numpy()[0:num_rec]
        func_rank = get_rank_func(num_rec)
        func = func_rank*user_utilities_ordered
        max_index = np.where(func == max(func))
        item_2_rec = int(recs.index[max_index][0])
        rating = recs.loc[item_2_rec]["predicted_ratings"]
        inter_users.append(user);inter_items.append(item_2_rec); inter_ratings.append(rating)
    
    interactions_by_iteration = [inter_users, inter_items, inter_ratings]
        
    return  interactions_by_iteration 


def add_users(num_u, simulation_step, new_u):
    if simulation_step!=0:
        for user in [*range(num_u + (simulation_step-1)*new_u + 1, num_u + (simulation_step)*new_u + 1,1)]:
            algo_cosin.add_user(user)
        return [*range(num_u + (simulation_step-1)*new_u + 1, num_u + (simulation_step)*new_u + 1,1)]
    else:
        return []
            
def add_items(num_i, simulation_step, new_i):
    if simulation_step!=0:
        for item in [*range(num_i + (simulation_step-1)*new_i + 1, num_i + (simulation_step)*new_i + 1,1)]:
            algo_cosin.add_item(item)
        return [*range(num_i + (simulation_step-1)*new_i + 1, num_i + (simulation_step)*new_i + 1,1)]
    else: 
        return[]
        
def add_interactions(list_of_interactions):
    algo_cosin.add_interactions(list_of_interactions[0], list_of_interactions[1], list_of_interactions[2])




num_u = len(movielens.users)
# num_u = 5
num_i = len(movielens.movies)
# num_i = 5
num_rec = 30
simulation_steps = 10
new_i = 10
new_u = 10


# Intializing teh full utilities matrix
StartTime = datetime.now()
print("Initizing Full Utility Matrix.......")
Full_Known_Utilities = initialize_known_utility(num_u, num_i, simulation_steps, new_i, new_u)
print("Initization of Full Utility Matrix Complete")
print("Time taken: ", datetime.now() - StartTime)


#storing max utilities by user for thresholding
# maximum_util_by_user = P_df.max(axis=1)
interactions = [[],[],[]]
for t in range(simulation_steps):
    StartTime = datetime.now()
    print("Starting iteration number: ", t+1)
    
    # interactions is arranged as a 3xn array of arrays, where n is the number of interactions. 
    # In intercations, dimension 1 is the users that interact, dimension 2 is the items that they interact with, dimension 3 is the ratings given.
    new_users = add_users(num_u, t, new_i)
    new_items = add_items(num_i, t, new_i)
    
    try:
        interactions[0].extend(new_users); interactions[1].extend(np.random.choice([*range(1,new_items[0],1)], size=(len(new_users)),replace=False)); interactions[2].extend(np.random.choice([1,2,3,4,5], size = (len(new_users))))
        interactions[0].extend(np.random.choice([*range(1,new_users[0],1)], size=(len(new_items)),replace=False)); interactions[1].extend(new_items); interactions[2].extend(np.random.choice([1,2,3,4,5], size = (len(new_items))))
    except IndexError:
        print('First itteration no new users or items added to interactions')
        
    
    add_interactions(interactions)
    interactions.append([t]*len(interactions[0]))
    interactions_df = pd.DataFrame(np.transpose(interactions), columns = ['user', 'item', 'rating', 'timestamp'])
    ratings = ratings.append(interactions_df, ignore_index=True)
    algo_cosin.fit(ratings)
    
    P_df_t = Full_Known_Utilities.iloc[0:(num_u+(t*new_u)),0:(num_i+(t*new_i))]
    
    # probability currently kept constant
    # change prob explore to a decreasing function to have a reudcing explore prob
    prob_explore = 0.4
    
    
    interactions = env_step(P_df_t, prob_explore,num_rec)
    
    print("Completed iteration number: ", t+1)
    print("Time taken: ", datetime.now() - StartTime)
   
    

