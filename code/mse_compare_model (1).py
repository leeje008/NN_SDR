import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import math

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F

import time
import datetime

import functools

from measure import *
from main import *
from data_generator import *

import ray


class Two_Network(nn.Module):
        def __init__(self, input_dim, iters, tolerance , epo, hidden_dim):
            super(Two_Network, self).__init__()
            self.input_dim = input_dim
            self.tolerance = tolerance
            self.hidden_dim = hidden_dim
            self.hidden_dim2 = 4
            self.learning_rate = 0.1
            self.epo = 10
            self.iters = iters
            
            # we have 3-hidden layer
            self.layer1 = nn.Linear(self.input_dim, self.input_dim) # input: 2D tensor and its layer equals to sufficient dimension reduction
            self.layer2 = nn.Linear(self.input_dim, self.input_dim) # neural network train
            self.layer3 = nn.Linear(self.input_dim, 1)
            
            self.bn1 = nn.BatchNorm1d(num_features = self.input_dim)
            self.bn2 = nn.BatchNorm1d(num_features = self.input_dim)
            
        def forward(self, X):
            X = F.tanh(self.bn1(self.layer1(X))) # tanh or sigmoid can be used
            X = F.sigmoid(self.bn2(self.layer2(X)))
            X = self.layer3(X)

            return X

                    
        def train(self, X, y, X_val, y_val):
            optimizer = optim.Adam( self.parameters(),  lr = self.learning_rate)
                
                
            loss_function = nn.MSELoss()
            # early_stopping = EarlyStopping(patience = 10, verbose = True)
           
            lr = self.learning_rate

            
            train_MSE_loss = [] # regularize loss 담을 거
            val_MSE_loss = []
            
            epochs = self.iters

            epo = self.epo

            
            p = X.shape[1]
            
            best_loss = 100.0
            
            PATH = "nonsdr_two_best_model.pt"
            early_stopping = EarlyStopping(patience = self.tolerance, verbose = True) # early stopping option
            
            for i in range(epochs + 1):                  
                
                # initialize the gradient of model parameters
                
                optimizer.zero_grad()
                
                # calculate the loss
                
                y_hat1 = self.forward(X)
                                
                # calculate the loss
                m_loss = loss_function(y_hat1, y)                                                                                                                                                                 
                m_loss.backward() # Computing Gradient (by GIST algorithm, we first calculate grdient of MSE Loss)                
                optimizer.step() # Backpropagation
                            
                                                               
                v1, tr_loss  = self.evaluate(X, y)

                v2, v_loss  = self.evaluate(X_val, y_val)
                
                                
                train_MSE_loss.append(v1)
                val_MSE_loss.append(v2)


                
                # best model save
                if v2 < best_loss:
                    best_loss = v2
                    torch.save(self.state_dict(), PATH)
                       
                if early_stopping.early_stop:
                    print(i)
                    print("Early stopping")
                    break
                    
            self.load_state_dict(torch.load(PATH)) # best parameter in train-process

                    
            return self, train_MSE_loss, val_MSE_loss
                    
        def predict(self, X_val):
            y_hat1 = self.forward(X_val)
            return y_hat1
            
        def evaluate(self, X_val, y_val):
            n_val = X_val.shape[0]
            y_hat2 = self.predict(X_val)
            e_val = (1/n_val) * torch.sum((y_val - y_hat2) **2).item()
            e_val_tensor = (1/n_val) * torch.sum((y_val - y_hat2) **2)
            return e_val , e_val_tensor
        
@ray.remote
class ray_Two_Network(nn.Module):
        def __init__(self, input_dim, iters, tolerance , epo, hidden_dim):
            super(Two_Network, self).__init__()
            self.input_dim = input_dim
            self.tolerance = tolerance
            self.hidden_dim = hidden_dim
            self.hidden_dim2 = 4
            self.learning_rate = 0.1
            self.epo = 10
            self.iters = iters
            
            # we have 3-hidden layer
            self.layer1 = nn.Linear(self.input_dim, self.input_dim) # input: 2D tensor and its layer equals to sufficient dimension reduction
            self.layer2 = nn.Linear(self.input_dim, self.input_dim) # neural network train
            self.layer3 = nn.Linear(self.input_dim, 1)
            
            self.bn1 = nn.BatchNorm1d(num_features = self.input_dim)
            self.bn2 = nn.BatchNorm1d(num_features = self.input_dim)
            
        def forward(self, X):
            X = F.tanh(self.bn1(self.layer1(X))) # tanh or sigmoid can be used
            X = F.sigmoid(self.bn2(self.layer2(X)))
            X = self.layer3(X)

            return X

                    
        def train(self, X, y, X_val, y_val):
            optimizer = optim.Adam( self.parameters(),  lr = self.learning_rate)
                
                
            loss_function = nn.MSELoss()
            # early_stopping = EarlyStopping(patience = 10, verbose = True)
           
            lr = self.learning_rate

            
            train_MSE_loss = [] # regularize loss 담을 거
            val_MSE_loss = []
            
            epochs = self.iters

            epo = self.epo

            
            p = X.shape[1]
            
            best_loss = 100.0
            
            PATH = "nonsdr_two_best_model.pt"
            early_stopping = EarlyStopping(patience = self.tolerance, verbose = True) # early stopping option
            
            for i in range(epochs + 1):                  
                
                # initialize the gradient of model parameters
                
                optimizer.zero_grad()
                
                # calculate the loss
                
                y_hat1 = self.forward(X)
                                
                # calculate the loss
                m_loss = loss_function(y_hat1, y)                                                                                                                                                                 
                m_loss.backward() # Computing Gradient (by GIST algorithm, we first calculate grdient of MSE Loss)                
                optimizer.step() # Backpropagation
                            
                                                               
                v1, tr_loss  = self.evaluate(X, y)

                v2, v_loss  = self.evaluate(X_val, y_val)
                
                                
                train_MSE_loss.append(v1)
                val_MSE_loss.append(v2)


                
                # best model save
                if v2 < best_loss:
                    best_loss = v2
                    torch.save(self.state_dict(), PATH)
                       
                if early_stopping.early_stop:
                    print(i)
                    print("Early stopping")
                    break
                    
            self.load_state_dict(torch.load(PATH)) # best parameter in train-process

                    
            return self, train_MSE_loss, val_MSE_loss
                    
        def predict(self, X_val):
            y_hat1 = self.forward(X_val)
            return y_hat1
            
        def evaluate(self, X_val, y_val):
            n_val = X_val.shape[0]
            y_hat2 = self.predict(X_val)
            e_val = (1/n_val) * torch.sum((y_val - y_hat2) **2).item()
            e_val_tensor = (1/n_val) * torch.sum((y_val - y_hat2) **2)
            return e_val , e_val_tensor
        
        
class Three_Network(nn.Module):
        def __init__(self, input_dim, iters, tolerance , epo, hidden_dim):
            super(Three_Network, self).__init__()
            self.input_dim = input_dim
            self.tolerance = tolerance
            self.hidden_dim = hidden_dim
            self.hidden_dim2 = 4
            self.learning_rate = 0.1
            self.epo = 10
            self.iters = iters
            
            # we have 3-hidden layer
            self.layer1 = nn.Linear(self.input_dim, self.input_dim) # input: 2D tensor and its layer equals to sufficient dimension reduction
            self.layer2 = nn.Linear(self.input_dim, self.input_dim) # neural network train
            self.layer3 = nn.Linear(self.input_dim, self.input_dim)
            self.layer4 = nn.Linear(self.input_dim, 1)
            
            self.bn1 = nn.BatchNorm1d(num_features = self.input_dim)
            self.bn2 = nn.BatchNorm1d(num_features = self.input_dim)
            self.bn3 = nn.BatchNorm1d(num_features = self.input_dim)

            
        def forward(self, X):
            X = F.tanh(self.bn1(self.layer1(X))) # tanh or sigmoid can be used
            X = F.sigmoid(self.bn2(self.layer2(X)))
            X = F.sigmoid(self.bn3(self.layer3(X)))
            X = self.layer3(X)

            return X

                    
        def train(self, X, y, X_val, y_val):
            optimizer = optim.Adam( self.parameters(),  lr = self.learning_rate)

                
                
            loss_function = nn.MSELoss()
            # early_stopping = EarlyStopping(patience = 10, verbose = True)
           
            lr = self.learning_rate

            
            train_MSE_loss = [] # regularize loss 담을 거
            val_MSE_loss = []
            
            epochs = self.iters

            epo = self.epo

            
            p = X.shape[1]
            
            best_loss = 100.0
            
            PATH = "nonsdr_three_best_model.pt"
            early_stopping = EarlyStopping(patience = self.tolerance, verbose = True) # early stopping option
            
            for i in range(epochs + 1):                  
                
                # initialize the gradient of model parameters
                
                optimizer.zero_grad()
                
                # calculate the loss
                
                y_hat1 = self.forward(X)
                                
                # calculate the loss
                m_loss = loss_function(y_hat1, y)                                                                                                                                                                 
                m_loss.backward() # Computing Gradient (by GIST algorithm, we first calculate grdient of MSE Loss)                
                optimizer.step() # Backpropagation
                            
                                                               
                v1, tr_loss  = self.evaluate(X, y)

                v2, v_loss  = self.evaluate(X_val, y_val)
                
                                
                train_MSE_loss.append(v1)
                val_MSE_loss.append(v2)

                
                # best model save
                if v2 < best_loss:
                    best_loss = v2
                    torch.save(self.state_dict(), PATH)
                       
                if early_stopping.early_stop:
                    print(i)
                    print("Early stopping")
                    break
                    
            self.load_state_dict(torch.load(PATH)) # best parameter in train-process

                    
            return self, train_MSE_loss, val_MSE_loss
                    
        def predict(self, X_val):
            y_hat1 = self.forward(X_val)
            return y_hat1
            
        def evaluate(self, X_val, y_val):
            n_val = X_val.shape[0]
            y_hat2 = self.predict(X_val)
            e_val = (1/n_val) * torch.sum((y_val - y_hat2) **2).item()
            e_val_tensor = (1/n_val) * torch.sum((y_val - y_hat2) **2)
            return e_val , e_val_tensor

