import pandas as pd
import numpy as np
import math

import os
from datetime import datetime
from algorithm.cosin import CosinSimilarity
import Datasets.Small_Test_Dataset as dataset


def getRatingsData():
    return pd.read_parquet(os.path.dirname(dataset.__file__)+'/ratings.parquet.gzip')

def getUtilityData():
    return pd.read_pickle(os.path.dirname(dataset.__file__)+'/utility_Matrix.pkl')


ratings = getRatingsData()
Full_Known_Utilities = getUtilityData()

algo_cosin = CosinSimilarity()

algo_cosin.fit(ratings)
   
    
def get_rank_func(recs, func_type = "sigmoid"):
    x = np.array([r for r in range(1,recs + 1,1)])
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
            algo_cosin.add_user(user)
            
def add_items(num_i, simulation_step, new_i):
    if simulation_step!=0:
        for item in [*range(num_i + (simulation_step-1)*new_i + 1, num_i + (simulation_step)*new_i + 1,1)]:
            algo_cosin.add_item(item)
        
def add_interactions(list_of_interactions):
    algo_cosin.add_interactions(list_of_interactions[0], list_of_interactions[1], list_of_interactions[2])

def get_rating(P_df_u_values, item):
    return round(1 + ((P_df_u_values[item]-P_df_u_values.min())*(5-1))/(P_df_u_values.max()-P_df_u_values.min()))
    


num_u = 200
# num_u = 5
num_i = 400
# num_i = 5
num_rec = 30
simulation_steps = 100
new_i = 2
new_u = 2


# # Intializing teh full utilities matrix
# StartTime = datetime.now()
# print("Initizing Full Utility Matrix.......")
# Full_Known_Utilities = initialize_known_utility(num_u, num_i, simulation_steps, new_i, new_u)
# print("Initization of Full Utility Matrix Complete")
# print("Time taken: ", datetime.now() - StartTime)


#storing max utilities by user for thresholding
# maximum_util_by_user = P_df.max(axis=1)
interactions = [[],[],[]]
for t in range(simulation_steps):
    StartTime = datetime.now()
    print("Starting iteration number: ", t+1)
    
    # interactions is arranged as a 3xn array of arrays, where n is the number of interactions. 
    # In intercations, dimension 1 is the users that interact, dimension 2 is the items that they interact with, dimension 3 is the ratings given.
    add_users(num_u, t, new_i)
    add_items(num_i, t, new_i)
    add_interactions(interactions)
    algo_cosin.update()
    
    P_df_t = Full_Known_Utilities.iloc[0:(num_u+(t*new_u)),0:(num_i+(t*new_i))]
    
    # probability currently kept constant
    # change prob explore to a decreasing function to have a reudcing explore prob
    prob_explore = 0.4
    
    
    interactions = env_step(P_df_t, prob_explore,num_rec)
    
    if np.isnan(sum(interactions[2])):
        print("There are nan ratings being added")
    
    print("Completed iteration number: ", t+1)
    print("Time taken: ", datetime.now() - StartTime)
   
    

