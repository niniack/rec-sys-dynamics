This file contains information on all the different datasets available in this directory. 

-----------------------------------------------------------------------------------------

creating_datasets.py is the python script for making the datasets, the variables are self explanatory in the script

For help accessing these datasets please go to the file accessing_datasets_help.py in the src folder

-----------------------------------------------------------------------------------------



-----------------------------------------------------------------------------------------

Every individual dataset is contained in a folder and has four useful files:
- ratings.parquet.gzip ----- The sparse representation of ratings
Column = user, item, rating, timestamp
Row = Interaction 


- Utility_Matrix.pkl ----- The full utility matrix for users and items including the items and users that will be added over time
Column = Item_Id
Row = User_Id

- P_df_Plot.png ----- This is a plot of the heat map of the dataset's complete utility matrix

- ratings_matrix_plot.png ----- This is a plot of the heat map of the dataset's initial utility matrix

-----------------------------------------------------------------------------------------



-----------------------------------------------------------------------------------------

Small_Test_Dataset
This dataset is for quick use testing. It contains equal bias on both sides


Number of Users: 200
Percentage of Left Biased Users = 10%
Percentage of Right Biased Users = 10%


Number of Items: 400
Percentage of Left Biased Items: 30%
Percentage of Right Biased Items: 30%

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 2
Number of Items Added Each Iteration 2

Total Initial Ratings = 10869/80000

-----------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------

Movie_Lens_Recreation
This is a replication of the movie lens dataset with biased communities 

Number of Users: 943
Number of Left Biased Users = 94
Number of Right Biased Users = 94


Number of Items: 1683
Number of Left Biased Items: 504
Number of Right Biased Items: 504

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 10
Number of Items Added Each Iteration 10

Total Initial Ratings = 100850/1587069
-----------------------------------------------------------------------------------------



