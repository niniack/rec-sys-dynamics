This file contains information on all the different datasets available in this directory. 

-----------------------------------------------------------------------------------------

creating_datasets.py is the python script for making the datasets, the variables are self explanatory in the script

For help accessing these datasets please go to the file accessing_datasets_help.py in the src folder

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
Number of Left Biased Users = 20
Number of Right Biased Users = 20


Number of Items: 400
Number of Left Biased Items: 120
Number of Right Biased Items: 120

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 2
Number of Items Added Each Iteration 2

Total Initial Ratings = 5046/80000

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








-----------------------------------------------------------------------------------------

All_Neutral
This is a replication of the movie lens dataset with biased communities 

Number of Users: 1000
Number of Left Biased Users = 0
Number of Right Biased Users = 0


Number of Items: 1700
Number of Left Biased Items: 0
Number of Right Biased Items: 0

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 10
Number of Items Added Each Iteration 10

Total Initial Ratings = 99827/1700000
-----------------------------------------------------------------------------------------








-----------------------------------------------------------------------------------------

1_Biased_Community_Control

Number of Users: 1000
Number of Left Biased Users = 100
Number of Right Biased Users = 0


Number of Items: 1700
Number of Left Biased Items: 510
Number of Right Biased Items: 0

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 10
Number of Items Added Each Iteration 10

Total Initial Ratings = 99323/1700000
Rating Ratio Biased:Unbiased User = 3.7:1
-----------------------------------------------------------------------------------------

1_Biased_Community_More_Users

Number of Users: 1000
Number of Left Biased Users = 200
Number of Right Biased Users = 0


Number of Items: 1700
Number of Left Biased Items: 510
Number of Right Biased Items: 0

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 10
Number of Items Added Each Iteration 10

Total Initial Ratings = 99723/1700000
Rating Ratio Biased:Unbiased User = 4:1
-----------------------------------------------------------------------------------------

1_Biased_Community_Less_Items

Number of Users: 1000
Number of Left Biased Users = 100
Number of Right Biased Users = 0


Number of Items: 1700
Number of Left Biased Items: 255
Number of Right Biased Items: 0

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 10
Number of Items Added Each Iteration 10

Total Initial Ratings = 100572/1700000
Rating Ratio Biased:Unbiased User = 3.5:1
-----------------------------------------------------------------------------------------








-----------------------------------------------------------------------------------------

2_Biased_Community_Control

Number of Users: 1000
Number of Left Biased Users = 100
Number of Right Biased Users = 100


Number of Items: 1700
Number of Left Biased Items: 510
Number of Right Biased Items: 510

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 10
Number of Items Added Each Iteration 10

Total Initial Ratings = 100398/1700000
Rating Ratio Biased:Unbiased User = 5:1
-----------------------------------------------------------------------------------------

2_Biased_Community_More_Users

Number of Users: 1000
Number of Left Biased Users = 150
Number of Right Biased Users = 150


Number of Items: 1700
Number of Left Biased Items: 510
Number of Right Biased Items: 510

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 10
Number of Items Added Each Iteration 10

Total Initial Ratings = 100496/1700000
Rating Ratio Biased:Unbiased User = 7:1
-----------------------------------------------------------------------------------------

2_Biased_Community_Less_Items 

Number of Users: 1000
Number of Left Biased Users = 100
Number of Right Biased Users = 100


Number of Items: 1700
Number of Left Biased Items: 255
Number of Right Biased Items: 255

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 10
Number of Items Added Each Iteration 10

Total Initial Ratings = 100401/1700000
Rating Ratio Biased:Unbiased User = 4:1
-----------------------------------------------------------------------------------------








-----------------------------------------------------------------------------------------

Biased_Neutral_Control

Number of Users: 1000
Number of Left Biased Users = 100
Number of Right Biased Users = 100


Number of Items: 1700
Number of Left Biased Items: 510
Number of Right Biased Items: 510

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 10
Number of Items Added Each Iteration 10

Total Initial Ratings = 99959/1700000
Rating Ratio Biased:Unbiased User = 5:1
-----------------------------------------------------------------------------------------

Biased_Neutral_More_Users

Number of Users: 1000
Number of Left Biased Users = 150
Number of Right Biased Users = 150


Number of Items: 1700
Number of Left Biased Items: 510
Number of Right Biased Items: 510

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 10
Number of Items Added Each Iteration 10

Total Initial Ratings = 98537/1700000
Rating Ratio Biased:Unbiased User = 7:1
-----------------------------------------------------------------------------------------

Biased_Neutral_Less_Items

Number of Users: 1000
Number of Left Biased Users = 100
Number of Right Biased Users = 100


Number of Items: 1700
Number of Left Biased Items: 255
Number of Right Biased Items: 255

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 10
Number of Items Added Each Iteration 10

Total Initial Ratings = 100353/1700000
Rating Ratio Biased:Unbiased User = 4:1
-----------------------------------------------------------------------------------------