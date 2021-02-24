# Import the libraries we will be using

import os
import numpy as np
import pandas as pd
import math
#import matplotlib.patches as patches
import matplotlib.pylab as plt

from sklearn import metrics
from sklearn import datasets

bookid = pd.read_csv("../datasets/UCSDbooks/book_id_map.csv")
userid = pd.read_csv("../datasets/UCSDbooks/user_id_map.csv")
interactions = pd.read_csv("../datasets/UCSDbooks/goodreads_interactions.csv")

