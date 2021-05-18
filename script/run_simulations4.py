import sys

### IMPORTANT ###
# Make sure you are correctly appending the path
# Otherwise the imports will not work!
sys.path.append("/Users/pvs262/Documents/rec-sys-dynamics/code")
from src.analysis.cluster import movielens, cluster, analysis, post_process
from src.analysis.simulate import simulate

##================================================================
## FOR PRAJNA
'''
# FOR All_Neutral
run = simulate('ease', 'All_Neutral')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
run_output = run.run_dynamics(n_i=10, n_u=0, n_r=30, steps=100, n_clusters = 2)
# save the plot_counts() and plot_percent pngs
analyse = analysis(run_output[1])
analyse.rename_cluster(1,1000)
analyse.plot_counts(show=False, loc=run.run_name+'/counts.png')
analyse.plot_percent(show=False, loc=run.run_name+'/percent.png')

# FOR 1_Biased_Community_Control
run = simulate('ease', '1_Biased_Communities_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
run_output = run.run_dynamics(n_i=10, n_u=0, n_r=30, steps=100, n_clusters = 2)
# save the plot_counts() and plot_percent pngs
analyse = analysis(run_output[1])
analyse.rename_cluster(1,1000)
analyse.plot_counts(show=False, loc=run.run_name+'/counts.png')
analyse.plot_percent(show=False, loc=run.run_name+'/percent.png')

# FOR 2_Biased_Community_Control
run = simulate('ease', '2_Biased_Communities_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
run_output = run.run_dynamics(n_i=10, n_u=0, n_r=30, steps=100, n_clusters = 3)
# save the plot_counts() and plot_percent pngs
analyse = analysis(run_output[1])
analyse.rename_cluster(1,1000)
analyse.plot_counts(show=False, loc=run.run_name+'/counts.png')
analyse.plot_percent(show=False, loc=run.run_name+'/percent.png')
'''
# FOR Biased_Neutral_Control
run = simulate('ease', 'Biased_Neutral_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
run_output = run.run_dynamics(n_i=10, n_u=0, n_r=30, steps=100, n_clusters = 3)
# save the plot_counts() and plot_percent pngs
analyse = analysis(run_output[1])
analyse.rename_cluster(1,1000)
analyse.plot_counts(show=False, loc=run.run_name+'/counts.png')
analyse.plot_percent(show=False, loc=run.run_name+'/percent.png')
'''
# FOR All_Neutral
run = simulate('mf', 'All_Neutral')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
run_output = run.run_dynamics(n_i=10, n_u=0, n_r=30, steps=100, n_clusters = 2)
# save the plot_counts() and plot_percent pngs
analyse = analysis(run_output[1])
analyse.rename_cluster(1,1000)
analyse.plot_counts(show=False, loc=run.run_name+'/counts.png')
analyse.plot_percent(show=False, loc=run.run_name+'/percent.png')

# FOR 1_Biased_Community_Control
run = simulate('mf', '1_Biased_Communities_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
run_output = run.run_dynamics(n_i=10, n_u=0, n_r=30, steps=100, n_clusters = 2)
# save the plot_counts() and plot_percent pngs
analyse = analysis(run_output[1])
analyse.rename_cluster(1,1000)
analyse.plot_counts(show=False, loc=run.run_name+'/counts.png')
analyse.plot_percent(show=False, loc=run.run_name+'/percent.png')

##================================================================
# FOR MANI

# FOR All_Neutral
run = simulate('cosin', 'All_Neutral')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
run_output = run.run_dynamics(n_i=10, n_u=0, n_r=30, steps=100, n_clusters = 2)
# save the plot_counts() and plot_percent pngs
analyse = analysis(run_output[1])
analyse.rename_cluster(1,1000)
analyse.plot_counts(show=False, loc=run.run_name+'/counts.png')
analyse.plot_percent(show=False, loc=run.run_name+'/percent.png')

# FOR 1_Biased_Community_Control
run = simulate('cosin', '1_Biased_Communities_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
run_output = run.run_dynamics(n_i=10, n_u=0, n_r=30, steps=100, n_clusters = 2)
# save the plot_counts() and plot_percent pngs
analyse = analysis(run_output[1])
analyse.rename_cluster(1,1000)
analyse.plot_counts(show=False, loc=run.run_name+'/counts.png')
analyse.plot_percent(show=False, loc=run.run_name+'/percent.png')

# FOR 2_Biased_Community_Control
run = simulate('cosin', '2_Biased_Communities_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
run_output = run.run_dynamics(n_i=10, n_u=0, n_r=30, steps=100, n_clusters = 3)
# save the plot_counts() and plot_percent pngs
analyse = analysis(run_output[1])
analyse.rename_cluster(1,1000)
analyse.plot_counts(show=False, loc=run.run_name+'/counts.png')
analyse.plot_percent(show=False, loc=run.run_name+'/percent.png')

# FOR Biased_Neutral_Control
run = simulate('cosin', 'Biased_Neutral_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
run_output = run.run_dynamics(n_i=10, n_u=0, n_r=30, steps=100, n_clusters = 3)
# save the plot_counts() and plot_percent pngs
analyse = analysis(run_output[1])
analyse.rename_cluster(1,1000)
analyse.plot_counts(show=False, loc=run.run_name+'/counts.png')
analyse.plot_percent(show=False, loc=run.run_name+'/percent.png')

# FOR 2_Biased_Community_Control
run = simulate('mf', '2_Biased_Communities_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
run_output = run.run_dynamics(n_i=10, n_u=0, n_r=30, steps=100, n_clusters = 3)
# save the plot_counts() and plot_percent pngs
analyse = analysis(run_output[1])
analyse.rename_cluster(1,1000)
analyse.plot_counts(show=False, loc=run.run_name+'/counts.png')
analyse.plot_percent(show=False, loc=run.run_name+'/percent.png')

# FOR Biased_Neutral_Control
run = simulate('mf', 'Biased_Neutral_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
run_output = run.run_dynamics(n_i=10, n_u=0, n_r=30, steps=100, n_clusters = 3)
# save the plot_counts() and plot_percent pngs
analyse = analysis(run_output[1])
analyse.rename_cluster(1,1000)
analyse.plot_counts(show=False, loc=run.run_name+'/counts.png')
analyse.plot_percent(show=False, loc=run.run_name+'/percent.png')
'''
