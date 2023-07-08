from data_generator import *
from init_sdr import *
from main import *
from common import *
from measure import *


import argparse
from itertools import product
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import datasets, transforms

import copy
import time
from sklearn.model_selection import KFold


import numpy as np
from typing import List, Dict
import warnings
import math
import random

from ray import tune


def train_model(config, state = 'test'):
    
    # set basic hyperparameter
    h_dim1 = config['h_dim1']
    h_dim2 = config['h_dim2']
    num_variables =  config["num_features"] 
    num_samples = config["sample_size"]  
    lambda_0 =  config["lambda_0"] 
    lambda_1 =  config.get('lambda_1', 0.1)   
    learning_rate =  config['lr']  
    epochs = config['epoch']  
    seed_val = config["seed"] 
    data_seed = config['data_seed']
    
    init_weight = config.get('init', None)
    data_path = config.get('path', None)
    batch = config.get('use_batch', False)
    or_pen = config.get('or_pen','pro')
    dim = config.get("dim", "unknown")
    
    model_type = config['model_num']
    
    
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(seed_val)
    random.seed(seed_val)
    
    
    device = torch.device("cpu")
    

    model = ray_SDR_net(input_dim = num_variables, h1_dim = h_dim1, h2_dim = h_dim2).to(device)
    criterion = nn.MSELoss()
    
    # define our dataset if data path is none then implement simulation dataset
    if data_path is None:
        if state == 'train':
            dataset = SyntheticDataset(sample_size = num_samples, num_features = num_variables, model_num= model_type, seed = data_seed)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_indices = [x for x in range(0, train_size)]
            
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset,  list(set(range(len(dataset))) - set(train_indices)))
        elif state == 'test':
            add_sample = num_samples + int(0.2 * num_samples)
            dataset = SyntheticDataset(sample_size = add_sample, num_features = num_variables, model_num= model_type, seed = data_seed)
            train_size = num_samples
            val_size = len(dataset) - train_size
            
            train_indices = [x for x in range(0, train_size)]
            
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset,  list(set(range(len(dataset))) - set(train_indices)))
    else:
        dataset = MyDataset(data_path)    
    
    train_loader = DataLoader(train_dataset, batch_size = train_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = val_size, shuffle=False)
    
    
    for train_x, train_y in train_loader:
        X, y = train_x, train_y
        
    # dr init and calculate covariance
    v_x = copy.deepcopy(torch.cov(X.T))
    model.apply(weights_init)

    # init value
    try:
         if num_variables >= h_dim1:
                if init_weight is None:
                    model.apply(weights_init)
                else:
                    init_tensor = init_weight
                    weight_dict = {'layer1.weight': copy.deepcopy(init_tensor.T)}
                    model.custom_weight_init(weight_dict, seed_val)
    except ValueError:
        print("Wrong Dimension")
        
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)    
    
    train_mse_list = []
    val_mse_list = []
    
    train_custom_list = []
    val_custom_list = []
                
    best_loss = float('inf')
    best_params = None

    for epoch in range(epochs):
        

        # print("Weight2 \n", model.state_dict()['layer1.weight'])
        
        for inputs, labels in train_loader:
            inputs, labels =  inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)  
        
        ortho_regularization = torch.tensor(0., requires_grad=True)
        lam0_penalty = torch.tensor(lambda_0, requires_grad=True)    
            
        for name, param in model.named_parameters():
            if 'layer1' in name and "weight" in name:
                weight = param.view(param.size(0), -1)
                if or_pen == 'pro':
                    ortho_regularization = ortho_regularization + torch.norm(torch.mm(weight, weight.t()) - torch.eye(weight.size(0)))**2
                    loss += lam0_penalty * ortho_regularization
                elif or_pen == 'orthogonal':
                    ortho_regularization = ortho_regularization + torch.norm(torch.mm(torch.mm(weight, v_x), weight.t()) - torch.eye(weight.size(0)))**2
                    loss += lam0_penalty * ortho_regularization
                else:
                    raise ValueError
                    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print("Gradient Updated", model.state_dict()['layer1.weight'])
        
        for name, param in model.named_parameters():
            if 'layer1' in name and "weight" in name:
                u_first_param = copy.deepcopy(param.data)

        if or_pen == 'pro':
            updated_penalty = pro_penalty(u_first_param, lambda_0)
        elif or_pen == 'orthogonal':
            updated_penalty = ortho_penalty(u_first_param, v_x ,lambda_0)
        else:
            raise ValueError
        # compute updated loss
        
        if dim == 'unknown':
            model.state_dict()['layer2.weight'] = GroupLasso_update_p(model.state_dict()['layer2.weight'],lambda_1, optimizer.param_groups[0]['lr'])
            penalty =  p_adj_pen(lambda_1, model.state_dict()['layer2.weight'])
            
            # calculate current loss
            
            with torch.no_grad():
                u_train_loss = 0.0
                u_valid_loss = 0.0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    
                    u_loss = criterion(outputs, labels)
                    u_train_loss += u_loss
                train_mse_list.append(u_train_loss.item())
                
                for val_inputs, val_outputs in val_loader:
                    val_inputs, val_outputs = val_inputs.to(device), val_outputs.to(device)
                    val_pred = model(val_inputs)
                    u_val_loss = criterion(val_pred, val_outputs)
                    u_valid_loss += u_val_loss
                val_mse_list.append(u_valid_loss.item())
                
            # if epoch % 500 == 0:
                # print("train_mse_loss:", u_train_loss.item())
                # print('val_mse_loss:', u_valid_loss.item())
                # print("Orthogonal Loss:", updated_penalty)
                # print("Penlaty on Second layer:", penalty.item())
                
            custom_loss = u_train_loss + updated_penalty + penalty
            cus_val_loss = u_valid_loss + updated_penalty + penalty
                    
            train_custom_list.append(custom_loss.item())
            val_custom_list.append(cus_val_loss.item())
            # print("custom loss", custom_loss.item())
        
            
            current_params = copy.deepcopy(model.state_dict())
            param_diff = l2_loss(current_params, best_params)
            lr_term = optimizer.param_groups[0]['lr'] * 0.5
            param_diff = param_diff * lr_term

            if train_custom_list[epoch] < best_loss - param_diff:
                best_loss = train_custom_list[epoch]
                best_params = copy.deepcopy(current_params)
            else:
                model.load_state_dict(best_params)
                optimizer.param_groups[0]['lr'] *= 0.9
                # print("lr decay")
                
            if optimizer.param_groups[0]['lr'] < 1e-20:
                # print('Epochs', epoch)
                # print('convergence')
                break
                
                
        elif dim == 'known':
            
            with torch.no_grad():
                u_train_loss = 0.0
                u_valid_loss = 0.0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    
                    u_loss = criterion(outputs, labels)
                    u_train_loss += u_loss
                train_mse_list.append(u_train_loss.item())
                
                for val_inputs, val_outputs in val_loader:
                    val_inputs, val_outputs = val_inputs.to(device), val_outputs.to(device)
                    val_pred = model(val_inputs)
                    u_val_loss = criterion(val_pred, val_outputs)
                    u_valid_loss += u_val_loss
                val_mse_list.append(u_valid_loss.item())
            
            custom_loss = u_train_loss + updated_penalty
            cus_val_loss = u_valid_loss + updated_penalty 
            
            train_custom_list.append(custom_loss.item())
            val_custom_list.append(cus_val_loss.item())
        else:
            raise ValueError

    return train_mse_list, val_mse_list, train_custom_list, val_custom_list, model