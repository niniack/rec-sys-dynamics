#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 17:25:08 2021

@author: ar4366
"""
import pandas as pd
import numpy as np
from scipy.stats import dirichlet
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt
import sys



def get_mu(v1, v2):
    mu = np.dot(v1,np.transpose(v2))
    if mu < 0.0000001:
        mu = 0.0000001
    if mu > 0.999999:
        mu = 0.999999
    return mu

def get_alpha_beta(mu, sigma=10**(-5)):
    a1 = ((1-mu)/(sigma**2)-(1/mu))*mu**2
    b1 = a1*((1/mu)- 1)
    return a1, b1

def get_Vui(u, i, p, a):
    global p_curr, a_curr
    B_mu = get_mu(p[u-1],a[i-1])
    alpha, beta = get_alpha_beta(B_mu)
    # if alpha<0 or beta<0:
    #     print(p[u-1])
    #     print(a[i-1])
    #     print(B_mu)
    #     print(alpha)
    #     print(beta)
    #     p_curr = p[u-1]
    #     a_curr = a[i-1]

    return float(beta_dist.rvs(alpha, beta))

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


ratings = pd.DataFrame(columns = ['user','item','rating','timestamp'])

number_lu = 20
number_ru = 20
number_lm = 120
number_rm = 120
prob_biased_rating = 0.2
prob_unbiased_rating = 0.045

K_pref_dimensions = 20
alpha_for_left = [100,100,100,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
alpha_for_right = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,100,100,100]
alpha_for_unbiased_user = [1]*20
alpha_for_unbiased_item = [100]*20
simulation_steps = 100
new_u = 2
new_i = 2


users = np.arange(200)
movies = np.arange(400)

if number_lu+number_ru>len(users):
    print('Error! User Proportion Exceed Total Number of Users.')
    sys.exit()
    
if number_lm+number_rm>len(movies):
    print('Error! Item Proportion Exceeds Total Number of Items.')
    sys.exit()
    
if type(number_lu) != int or type(number_ru) != int or type(number_rm) != int or type(number_lm) != int:
    print('Error! User or Item Proportions Are Not Integers As Expected By The Script.')
    sys.exit()

data = np.zeros((len(users), len(movies)))
data_1 = np.zeros(((len(users)+(new_u*simulation_steps)),(len(movies)+(new_i*simulation_steps))))

ratings_df = pd.DataFrame(data, index=np.arange(1,len(users)+1,1), columns=np.arange(1,len(movies)+1,1))
V_df = pd.DataFrame(data_1, index=np.arange(1,(len(users)+(new_u*simulation_steps))+1,1), columns=np.arange(1,(len(movies)+(new_i*simulation_steps))+1,1))
P_df = pd.DataFrame(data_1, index=np.arange(1,(len(users)+(new_u*simulation_steps))+1,1), columns=np.arange(1,(len(movies)+(new_i*simulation_steps))+1,1))


u_p_left = dirichlet.rvs(alpha_for_left, size=1)
u_p_right = dirichlet.rvs(alpha_for_right, size=1)
u_p_unbiased = dirichlet.rvs(alpha_for_unbiased_user, size=1)
u_a_left = dirichlet.rvs(alpha_for_left, size=1)
u_a_right = dirichlet.rvs(alpha_for_right, size=1)
u_a_unbiased = dirichlet.rvs(alpha_for_unbiased_item, size=1)

p_left = dirichlet.rvs(u_p_left[0], size=number_lu)
p_right = dirichlet.rvs(u_p_right[0], size=number_ru)
p_unbiased = dirichlet.rvs(u_p_unbiased[0]*10, size=len(users)-number_lu-number_ru)
p_new = dirichlet.rvs(u_p_unbiased[0]*10, size=new_u*simulation_steps)
p = np.concatenate((p_left, p_unbiased,p_right,p_new), axis=0)

a_left = dirichlet.rvs(u_a_left[0], size=number_lm)
a_right = dirichlet.rvs(u_a_right[0], size=number_rm)
a_unbiased = dirichlet.rvs(u_a_unbiased[0]*0.1, size=len(movies)-number_lm-number_rm)
a_new = dirichlet.rvs(u_a_unbiased[0]*0.1, size=new_i*simulation_steps)
a = np.concatenate((a_left, a_unbiased,a_right,a_new), axis = 0)

for u in np.arange(1,(len(users)+(new_u*simulation_steps))+1,1):
    for m in np.arange(1, (len(movies)+(new_i*simulation_steps))+1,1):
        Vui = get_Vui(u,m, p, a)
        V_df.loc[u][m] = Vui
        known_proportion = get_nui()
        P_df.loc[u][m] = Vui*known_proportion
        if m<= len(movies) and u<= len(users):
            if u <= number_lu and m <= number_lm:
                if prob_biased_rating >= np.random.rand():
                    r = np.random.choice([4,5])
                    ratings_df.loc[u,m] = r
                    ratings = ratings.append({'user':u,'item':m, 'rating':r, 'timestamp':0}, ignore_index=True)
            elif u <= number_lu and m > len(movies)-number_rm:
                if prob_biased_rating >= np.random.rand():
                    r = np.random.choice([1,2])
                    ratings_df.loc[u,m] = r
                    ratings = ratings.append({'user':u,'item':m, 'rating':r, 'timestamp':0}, ignore_index=True)
            elif u > len(users)-number_ru and m <= number_lm:
                if prob_biased_rating >= np.random.rand():
                    r = np.random.choice([1,2])
                    ratings_df.loc[u,m] = r
                    ratings = ratings.append({'user':u,'item':m, 'rating':r, 'timestamp':0}, ignore_index=True)
            elif u > len(users)-number_ru and m > len(movies)-number_rm:
                if prob_biased_rating >= np.random.rand():
                    r = np.random.choice([4,5])
                    ratings_df.loc[u,m] = r
                    ratings = ratings.append({'user':u,'item':m, 'rating':r, 'timestamp':0}, ignore_index=True)
            else:
                if prob_unbiased_rating >= np.random.rand():
                    r = np.random.choice([1,2,3,4,5])
                    ratings_df.loc[u,m] = r
                    ratings = ratings.append({'user':u,'item':m, 'rating':r, 'timestamp':0}, ignore_index=True)
               
            




