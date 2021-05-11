import sys

### IMPORTANT ###
# Make sure you are correctly appending the path
# Otherwise the imports will not work!
sys.path.append("/Users/pvs262/Documents/rec-sys-dynamics/code")
from src.analysis.cluster import movielens, cluster, analysis, post_process
from src.analysis.simulate import simulate

##================================================================
## FOR PRAJNA

# FOR All_Neutral
test1 = simulate('ease', 'All_Neutral')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
test1_output = test1.run_dynamics(n_i=5, n_u=0, n_r=30, steps=100, n_clusters = 2)

# FOR 1_Biased_Community_Control
test1 = simulate('ease', '1_Biased_Community_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
test1_output = test1.run_dynamics(n_i=5, n_u=0, n_r=30, steps=100, n_clusters = 2)

# FOR 2_Biased_Community_Control
test1 = simulate('ease', '2_Biased_Community_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
test1_output = test1.run_dynamics(n_i=5, n_u=0, n_r=30, steps=100, n_clusters = 3)

# FOR Biased_Neutral_Control
test1 = simulate('ease', 'Biased_Neutral_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
test1_output = test1.run_dynamics(n_i=5, n_u=0, n_r=30, steps=100, n_clusters = 2)

# FOR All_Neutral
test1 = simulate('mf', 'All_Neutral')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
test1_output = test1.run_dynamics(n_i=5, n_u=0, n_r=30, steps=100, n_clusters = 2)

# FOR 1_Biased_Community_Control
test1 = simulate('mf', '1_Biased_Community_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
test1_output = test1.run_dynamics(n_i=5, n_u=0, n_r=30, steps=100, n_clusters = 2)

##================================================================
## FOR MANI

# FOR All_Neutral
test1 = simulate('cosin', 'All_Neutral')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
test1_output = test1.run_dynamics(n_i=5, n_u=0, n_r=30, steps=100, n_clusters = 2)

# FOR 1_Biased_Community_Control
test1 = simulate('cosin', '1_Biased_Community_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
test1_output = test1.run_dynamics(n_i=5, n_u=0, n_r=30, steps=100, n_clusters = 2)

# FOR 2_Biased_Community_Control
test1 = simulate('cosin', '2_Biased_Community_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
test1_output = test1.run_dynamics(n_i=5, n_u=0, n_r=30, steps=100, n_clusters = 3)

# FOR Biased_Neutral_Control
test1 = simulate('cosin', 'Biased_Neutral_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
test1_output = test1.run_dynamics(n_i=5, n_u=0, n_r=30, steps=100, n_clusters = 2)

# FOR 2_Biased_Community_Control
test1 = simulate('mf', '2_Biased_Community_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
test1_output = test1.run_dynamics(n_i=5, n_u=0, n_r=30, steps=100, n_clusters = 2)

# FOR Biased_Neutral_Control
test1 = simulate('mf', 'Biased_Neutral_Control')
#simulate.run_dynamics(n_i, n_u, n_r, steps)
test1_output = test1.run_dynamics(n_i=5, n_u=0, n_r=30, steps=100, n_clusters = 2)
