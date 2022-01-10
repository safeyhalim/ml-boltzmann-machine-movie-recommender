# Importing libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn # PyTorch neural network module
import torch.nn.parallel # PyTorch neural network parallel computing module
import torch.optim as optim # PyTorch optimizer
import torch.utils.data # PyTorch utilities
from torch.autograd import Variable # PyTorch gradient descent module

# Importing the dataset
# Note: the separator of the dataset is double colon '::'
# The dataset file has no header row (that's why the header is set to None)
# engine is set to python (the default is 'c'), because the python engine is more complete
# (has all the features while the c engine doesn't). This ensures that the dataset is correctly imported
# encoding: latin-1 because some of the movies have some special characters (the default is utf-8)
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the testing set
# The 100k dataset is used for the training/testing sets. 
# The MovieLens 100k dataset is already split into 5 training/testing set pairs u1 to u5
# (each is a diffrent splits of the 100k ratings) to allow for a 5 fold cross-validation analysis
# For this implementation, we are not interested in the cross validation, that's why we will just 
# use the training and the testing sets of u1
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int') # converting the training_set from a dataframe into array, because we are going to use PyTorch tensors (which expect arrays)
# Same for the test set
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the total number of users and movies in the dataset
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0]))) # the total number of users is the highest user ID (column 0) either in the training set or the testing set (because both the training and the testing sets have all the number of users: IDs from 1 to max)
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1]))) # Same logic applies for the total number of movies

