# Importing libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn # PyTorch neural network module
import torch.nn.parallel # PyTorch neural network parallel computing module
import torch.optim as optim # PyTorch optimizer
import torch.utils.data # PyTorch utilities
from torch.autograd import Variable # PyTorch gradient descent module

# This Recommender System based on Restricted Blotzmann Machines will predict whether a user likes a movie or not (Yes/No prediction)
# It doesn't predict the actual rating of the user for the Movie

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

# Rbm expects a matrix of observations as an input. Therefore, we are transforming the training set and the testing set into two matrices, each contains all the users and all the movies
# The rows of each matrix are the users, and the columns are the movies. Each cell of the matrix corresponds to a rating of 
# this user to this movie. If the user doesn't have a rating for a particular movie, a 0 is inserted in this cell.
def convert(data):
    new_data = []
    for id_user in range(1, nb_users + 1): # user Ids start at 1 (not 0)
        id_movies = data[:, 1][data[:, 0] == id_user] # take all the movie ids (column 1), such that the corresponding user Id (column 0) is equal to this user Id
        user_ratings = data[:, 2][data[:, 0] == id_user] # take the ratings all the ratings of this user
        ratings = np.zeros(nb_movies) # To make sure to have zeros for the movies that the user hasn't rated, we initialize an array of zeros 
        ratings[id_movies - 1] = user_ratings # Then replace the zeros with the user's ratings in the right indices (note the -1 because the arrays are indexed starting with 0, but the movies indices start with 1)
        new_data.append(list(ratings)) # Adding this user's movie ratings to the matrix
    return new_data

# Applying to the convert method to the training set and the testing set
training_set = convert(training_set)
test_set = convert(test_set)


# Converting the data into PyTorch Tensors
# Tensors are arrays that contain elements of a single data type. It's a multidimensional array (matrix). 
# We can still use Numpy arrays for the same purpose, but PyTorch Tensors are more efficient (Also more efficient than TensorFlow Tensors)
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) and 0 (Not Liked) --> because this is what the Rbm will predict
training_set[training_set == 0] = -1 # Unrated movies should have a different value than 0. This line takes all the values of training_set torch which are equal to zero and replace them
# with -1 (Why we didn't set the unrated movies to -1 from the beginning?! I don't know :))
# Not liked movies will be the movies that were rated either 1 or 2 (note: in PyTorch, there is no "or" operator, that's why we have to write two separate lines)
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
# Movies rated 3 or above, will be considered Liked by the user
training_set[training_set >= 3] = 1

# Do the same for the testing set
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
# The Restricted Boltzmann Machine is a Probabilistic Graphical Model
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) # Weights represents the probabilities of the visible nodes given the hidden nodes.
        # They are randomly initialized (following a normal distribution) as a Torch tensor (matrix) with the dimensions nhxnv (number of hidden nodes x number of visible nodes)
        self.a = torch.randn(1, nh) # The bias of the probabilities of hidden nodes given the visible nodes. Randomly initialized (following a normal distribution).
        # as a Torch tensor: one dimension represents the batch, and the second is for the hidden nodes themselves
        self.b = torch.randn(1, nv) # The bias of the probabilities of visible nodes given the hidden nodes (same idea as the previous bias)

    # Sampling the hidden nodes according to the probability p(h|v) --> p(h) given v, where h is a hidden node and v is a visible node
    # This probability p(h|v) is the Sigmoid activation function. 
    # Why do we need this sampling of h function? because, during the training, we will approximate the log likelihood gradient using Gibbs sampling.
    # And to apply the Gibbs sampling, we need to calculate the probability of the activation of the hidden nodes given the visible nodes.
    # What the function will do exactly is that it will activate some of the hidden nodes according to a certain probability p(h|v) (the probability that the hidden neuron equals 1 given the value of the visible neurons, which is the input vector of observations) that will be computed in the same function
    # It will return a vector of activation probabilities for each of the hidden nodes, and a vector of binary values for each of the hidden nodes depending on which of them is actually activated (1 is activated, 0 is not)
    # x: represents the visible neurons v in the probability p(h|v)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t()) # Multiplying the visibile neurons and the matrix of weights
        activation = wx + self.a.expand_as(wx) # a will be expanded to be of the same size as wx
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v) # We are actually buidling a Bernoulli Restricted Boltzmann Machine, which means that the machine should predict binary values (like a movie or dislike a movie).
        # p_h_given_v is a vector that represents for each hidden node the probability of it to be activated. The Torch Bernoulli method will return for each node of this vector whether it will actually be activated or not according to this probability (a vector of 0's and 1's)

    # Same as the method above, but here we are sampling the visible nodes given the hidden nodes p(v|h)
    # y: represents the hidden neurons h in the probability p(v|h)
    def sample_v(self, y):
        wy = torch.mm(y, self.W) # We shouldn't do the transpose here because we are multiplying by the vector of the hidden nodes
        activation = wy + self.b.expand_as(wy) # Multiplying by the bias of the visible nodes
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    
    # Approximating the RBM Log-Likelihood Gradient using Contrastive Divergence
    # We are trying to minimize an energy function: we ultimately want to optimize the weights to obtain minimum energy
    # This minimization of the energey is equivalent to maximizing the Log-Likelihood of the training set. To achieve
    # that, we need to compute the gradient. The computation of the gradient is too heavy, therefore we will try instead to
    # approximate it. Each time, we make tiny adjustments in the right direction (the direction of minimizing the energy). The algorithm that
    # will allow us to do these approximations is Contrastive Divergence which comes with Gibbs Sampling.
    # Gibbs Sampling consists of creating Gibb Chain in k steps (sampling the hidden nodes and the visible nodes k times).
    # We do that by initially taking the input vector (visible nodes): v(0) and sample the hidden nodes P(h|v), and then use the output
    # to sample the visible nodes in the second iteration v(1) using P(v|h), and so on. We do that k times.
    # v0: the visible nodes at the beginning (input ratings)
    # vk: the visible nodes after k samplings
    # ph0: the probabilities of the hidden nodes equal 1 given the initial visible nodes v0
    # phk: the probabilities of the hidden nodes equal 1 given the visible nodes vk (after k samplings)
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t() # Update the matrix of weights
        self.b += torch.sum((v0 - vk), 0) # Updating the bias of the probabilities of visible nodes given hidden nodes. We are adding 0 with this torch sum function just to keep the correct torch format.
        self.a += torch.sum((ph0 - phk), 0) # Updating the bias of probabilities of the hidden nodes given the visible nodes


# Creating the RBM model
nv = len(training_set[0]) # Number of visible nodes: the number of movies in the training set (the row length of the training set tensor)
nh = 100 # The number of hidden nodes (the number of features that we want the RBM to detect). We can start with any number here: we just choose a reasonable number that might be equivalent
# to the feature relevant to movies, for example: director, genre, winning oscar, etc. Can be tuned later on to optimize the model
batch_size = 100 # The batch size of observations to feed to the RBM in one iteration. Weights of the RBM will change after each batch. This is more efficient than feeding the observations one by one. Can be tuned later to improve the model
rbm = RBM(nv, nh)

# Training
nb_epoch = 10 # Chosen at random
for epoch in range(1, nb_epoch + 1):
    train_loss_average_distance = 0
    train_loss_rmse = 0
    s = 0. # counter, which will be used to normalize the train loss
    for user_id in range(0, nb_users - batch_size, batch_size): # The third argument in range is the loop step (every step is a batch size)
        vk = training_set[user_id : user_id + batch_size]
        v0 = training_set[user_id : user_id + batch_size] # This is the target observation that we will be comparing the prediction against to calculate the train loss
        ph0, _ = rbm.sample_h(v0)
        # Starting the Random Walk for the Gibbs Sampling
        for k in range(10): # We chose k = 10. Why?!
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0] # We want to make sure that the visible nodes with the values -1 (the movies that weren't rated by the users) keep their -1, so that they don't enter the training phase
            # So in vk, we set the values where their corresponding indices in v0 is -1 to -1
        phk, _ = rbm.sample_h(vk)
        # train
        rbm.train(v0, vk, ph0, phk) # Update the weights and the biases
        # Calculating the train loss using the Mean Absolute Distance between the target visible nodes and the predicted visible nodes
        train_loss_average_distance += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0])) # We are excluding the unrated visible nodes (where values equal -1)
        train_loss_rmse += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # Calculating the train loss with RMSE
        s += 1. # incrementing the counter
    print('epoch: ' + str(epoch) + ' loss (Average Distance): ' + str(train_loss_average_distance/s)) # Printing the normalized train loss for the current epoch (Average Distance)
    print('epoch: ' + str(epoch) + ' loss (rmse): ' + str(train_loss_rmse/s)) # Printing the normalized train loss for the current epoch (RMSE)


# Testing the RBM
# Blind Walk Markov Chain Monte Carlo (MCMC) Technique
test_loss_average_distance = 0
test_loss_rmse = 0
s = 0.
for user_id in range(nb_users):
    v = training_set[user_id : user_id + 1] # Input based on which we are making the predictions (The ratings of the current user in the training set). We are using the inputs of the training set to activate the neurons of the RBM to get the predictions for the unrated movies in the test set
    vt = test_set[user_id : user_id + 1] # Target (ratings of the current user in the test set) with which we are comparing the prediction result
    if len(vt[vt >= 0]) > 0: # We do it only if the current user actually has ratings in the test set
        _, h = rbm.sample_h(v) # During the training, we did k steps random walk, but for testing (running the model), it's sufficient to do only one step just to fire the hidden nodes based on the input ratings of the current user
        _, v = rbm.sample_v(h)
        test_loss_average_distance += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0])) # We are interested in cacluating the loss (prediction error) only for the movies that have ratings by the current user
        test_loss_rmse += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE
        s += 1.
print('loss (Average Distance): ' + str(test_loss_average_distance/s)) # Averaging the test loss (error) for predictions of all users (Average Distance)
print('loss (RMSE): ' + str(test_loss_rmse/s))
