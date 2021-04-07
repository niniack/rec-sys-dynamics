This file contains information on all the different datasets available in this directory. 

-----------------------------------------------------------------------------------------

creating_datasets.py is the python script for making the datasets, the variables are self explanatory in the script

-----------------------------------------------------------------------------------------



-----------------------------------------------------------------------------------------

Every individual dataset is contained in a folder and has two files:
- ratings.parquet.gzip ----- The sparse representation of ratings
Column = user, item, rating, timestamp
Row = Interaction 


- Utility_Matrix.pkl ----- The full utility matrix for users and items including the items and users that will be added over time
Column = Item_Id
Row = User_Id

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

Number of Users: 963
Percentage of Left Biased Users = 10%
Percentage of Right Biased Users = 10%


Number of Items: 1682
Percentage of Left Biased Items: 30%
Percentage of Right Biased Items: 30%

Total Number of Running Iterations: 100
Number of Users Added Each Iteration: 2
Number of Items Added Each Iteration 2

Total Initial Ratings = 10869/80000
-----------------------------------------------------------------------------------------

