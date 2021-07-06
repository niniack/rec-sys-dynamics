# source: https://youtu.be/-Mx89Jcn2E4

import pandas as pd  # (version 1.0.0)
import plotly.express as px  # (version 4.7.0)
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
#import sys

#sys.path.append("/Users/pvs262/Documents/rec-sys-dynamics/code/examples")

# No Explore
#folders = ['cosin_AN_0_0_30_100','cosin_BNC_0_0_30_100','cosin_1BCC_0_0_30_100','cosin_1BCMU_0_0_30_100','cosin_1BCLI_0_0_30_100','cosin_2BCC_0_0_30_100','cosin_2BCMU_0_0_30_100','cosin_2BCLI_0_0_30_100','mf_AN_0_0_30_100','mf_BNC_0_0_30_100','mf_1BCC_0_0_30_100','mf_1BCMU_0_0_30_100','mf_1BCLI_0_0_30_100','mf_2BCC_0_0_30_100','mf_2BCMU_0_0_30_100','mf_2BCLI_0_0_30_100']
#folders = ['item_based_AN_0_0_30_100','item_based_BNC_0_0_30_100','item_based_1BCC_0_0_30_100','item_based_1BCMU_0_0_30_100','item_based_1BCLI_0_0_30_100','item_based_2BCC_0_0_30_100','item_based_2BCMU_0_0_30_100','item_based_2BCLI_0_0_30_100']

# Explore
#folders = ['cosin_AN_10_0_30_100','cosin_BNC_10_0_30_100','cosin_1BCC_10_0_30_100','cosin_1BCMU_10_0_30_100','cosin_1BCLI_10_0_30_100','cosin_2BCC_10_0_30_100','cosin_2BCMU_10_0_30_100','cosin_2BCLI_10_0_30_100','mf_AN_10_0_30_100','mf_BNC_10_0_30_100','mf_1BCC_10_0_30_100','mf_1BCMU_10_0_30_100','mf_1BCLI_10_0_30_100','mf_2BCC_10_0_30_100','mf_2BCMU_10_0_30_100','mf_2BCLI_10_0_30_100']
#folders = ['item_based_AN_10_0_30_100','item_based_BNC_10_0_30_100','item_based_1BCC_10_0_30_100','item_based_1BCMU_10_0_30_100','item_based_1BCLI_10_0_30_100','item_based_2BCC_10_0_30_100','item_based_2BCMU_10_0_30_100','item_based_2BCLI_10_0_30_100']

folders = ['item_based_2BCLI_0_0_30_100']

for run in folders:

    df = pd.read_csv('datasets/syn_adjusted/No_Explore_4/'+run+'.csv')

    fig = px.scatter_3d(
        data_frame=df,
        x='0',
        y='1',
        z='2',
        color="cluster",
        color_discrete_sequence=['magenta', 'green', 'blue'],
        color_discrete_map={'0': 'green', '1': 'magenta', '2': 'blue'},

        title=run,
   
        height=700,                 # height of graph in pixels

        animation_frame='iteration',   # assign marks to animation frames
        range_x=[df['0'].min(),df['0'].max()],
        range_z=[df['1'].min(),df['1'].max()],
        range_y=[df['2'].min(),df['2'].max()]
    )

    fig.write_html("results/animation_small/adjusted/No_Explore/"+run+".html", include_plotlyjs='cdn')

pio.show(fig)


