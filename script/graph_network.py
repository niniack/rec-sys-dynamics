#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:26:34 2021

@author: ar4366
"""
import pandas as pd
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

### IMPORTANT ###
# Make sure you are correctly appending the path
# Otherwise the imports will not work!
def get_similarity(df):
    similarity_matrix = pd.DataFrame(data=cosine_similarity(df), index = np.arange(1,len(df)+1), columns = np.arange(1,len(df)+1))
    return similarity_matrix

def get_node_degree(sim_mat):
    node_degree = sim_mat.sum()
    node_degree = node_degree.subtract(pd.Series(data = np.ones(len(sim_mat)), index= np.arange(1,len(sim_mat)+1)))
    node_degree = node_degree
    # node_degree.sort_values(ascending = True, inplace = True)
    return node_degree

def get_bins(max_val, min_val):
    val = (max_val-min_val)**(1/20)
    bin_edges = [min_val+val**i for i in range(1,21)]
    bin_edges.append(min_val)
    bin_edges.sort()
    return bin_edges
    
    
def plot_node_degree(w_run_type, w_run, log_x=False, log_y=False):
    global node_degree_df
    plt.figure(); node_degree_df[w_run_type][w_run].plot.hist(logx=log_x, logy=log_y, bins=30, histtype="step",cumulative=True); plt.title(w_run_type + ' - ' + w_run); plt.xlabel('Node Degree'); plt.ylabel('f(Node Degree)');plt.xlim(0,600);plt.ylim(0,1000); plt.savefig(w_run_type + '_' + w_run + '.pdf')
    
os.chdir('../')
os.chdir('simulation_runs/Node_Degree_Plots')    

dir_path = "/Users/ar4366/Documents/GitHub/rec-sys-dynamics/simulation_runs/"
raw_data_frames = {}
node_degree_df = {}

for run_type in ["No_Explore_4", "Explore_Threshold_2"]:
    if run_type == "No_Explore_4":
        raw_runs = {}
        nd_runs = {}
        for run in ["cosin_1BCC_0_0_30_100", "cosin_1BCLI_0_0_30_100", "cosin_1BCMU_0_0_30_100", "cosin_2BCC_0_0_30_100", "cosin_2BCLI_0_0_30_100", "cosin_2BCMU_0_0_30_100", "cosin_AN_0_0_30_100", "cosin_BNC_0_0_30_100", "mf_1BCC_0_0_30_100", "mf_1BCLI_0_0_30_100", "mf_1BCMU_0_0_30_100", "mf_2BCC_0_0_30_100", "mf_2BCLI_0_0_30_100", "mf_2BCMU_0_0_30_100", "mf_AN_0_0_30_100", "mf_BNC_0_0_30_100"]:
            file_path = dir_path + run_type + '/' + run + "/UI0pkl.gzip"
            raw_runs[run] = pd.read_pickle(file_path, compression='gzip')
            nd_runs[run] = get_node_degree(get_similarity(pd.read_pickle(file_path, compression='gzip')))
        raw_data_frames[run_type] = raw_runs 
        node_degree_df[run_type] = nd_runs
    if run_type == "Explore_Threshold_2":
        raw_runs = {}
        nd_runs = {}
        for run in ["cosin_1BCC_10_0_30_100", "cosin_1BCLI_10_0_30_100", "cosin_1BCMU_10_0_30_100", "cosin_2BCC_10_0_30_100", "cosin_2BCLI_10_0_30_100", "cosin_2BCMU_10_0_30_100", "cosin_AN_10_0_30_100", "cosin_BNC_10_0_30_100", "mf_1BCC_10_0_30_100", "mf_1BCLI_10_0_30_100", "mf_1BCMU_10_0_30_100", "mf_2BCC_10_0_30_100", "mf_2BCLI_10_0_30_100", "mf_2BCMU_10_0_30_100", "mf_AN_10_0_30_100", "mf_BNC_10_0_30_100"]:
            file_path = dir_path + run_type + '/' + run + "/UI0pkl.gzip"
            raw_runs[run] = pd.read_pickle(file_path, compression='gzip')
            nd_runs[run] = get_node_degree(get_similarity(pd.read_pickle(file_path, compression='gzip')))
        raw_data_frames[run_type] = raw_runs 
        node_degree_df[run_type] = nd_runs


for x in ["No_Explore_4", "Explore_Threshold_2"]:
    if x == "No_Explore_4":
        for y in ["cosin_1BCC_0_0_30_100", "cosin_1BCLI_0_0_30_100", "cosin_1BCMU_0_0_30_100", "cosin_2BCC_0_0_30_100", "cosin_2BCLI_0_0_30_100", "cosin_2BCMU_0_0_30_100", "cosin_AN_0_0_30_100", "cosin_BNC_0_0_30_100", "mf_1BCC_0_0_30_100", "mf_1BCLI_0_0_30_100", "mf_1BCMU_0_0_30_100", "mf_2BCC_0_0_30_100", "mf_2BCLI_0_0_30_100", "mf_2BCMU_0_0_30_100", "mf_AN_0_0_30_100", "mf_BNC_0_0_30_100"]:
            plot_node_degree(x,y)
    if x == "Explore_Threshold_2":
        for y in ["cosin_1BCC_10_0_30_100", "cosin_1BCLI_10_0_30_100", "cosin_1BCMU_10_0_30_100", "cosin_2BCC_10_0_30_100", "cosin_2BCLI_10_0_30_100", "cosin_2BCMU_10_0_30_100", "cosin_AN_10_0_30_100", "cosin_BNC_10_0_30_100", "mf_1BCC_10_0_30_100", "mf_1BCLI_10_0_30_100", "mf_1BCMU_10_0_30_100", "mf_2BCC_10_0_30_100", "mf_2BCLI_10_0_30_100", "mf_2BCMU_10_0_30_100", "mf_AN_10_0_30_100", "mf_BNC_10_0_30_100"]:
            plot_node_degree(x,y)
    


