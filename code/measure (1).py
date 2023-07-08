import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import math

from sklearn import preprocessing
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

def nonzero_dim(W, THRES):
    non_zero = torch.max(torch.abs(W), dim = 1).values.numpy() # extract non-zero column by checking max value
    non_zero_index = np.where(non_zero >  THRES) # non_zero index
    y = [items for items in non_zero_index[0]] # transformation of non_zero_index
    return y


def nonzero_dim1(W, THRES):
    non_zero = torch.max(torch.abs(W), dim = 0).values.numpy() # extract non-zero column by checking max value
    non_zero_index = np.where(non_zero >  THRES) # non_zero index
    y = [items for items in non_zero_index[1]] # transformation of non_zero_index
    return y

def pow_cov(X, p):
    L, V = torch.linalg.eig(X) # eigen value decompostion 
    
    LL = torch.tensor(L, dtype = torch.float32)
    
    LL = torch.where(LL > 1e-5, LL, 0.0)# small or negative eigenvalue replace with zero
    
    LL = torch.tensor(LL, dtype = torch.complex64)
                
    pow_L = torch.pow(LL,p) # apply pow function, zero output nan
    diag = torch.nan_to_num(torch.tensor(torch.diag_embed(pow_L),dtype = torch.float32)) # nan replace with 0
    diag = torch.tensor(diag, dtype = torch.complex64) # change dtype with complex 
    pow_cov = V @ diag @ torch.linalg.inv(V)
    pow_cov = torch.tensor(pow_cov,  dtype=torch.float32)
    return pow_cov


def multiple_corr(W, beta, X ,mode = 'predictor'):
    est = W[nonzero_dim(W, 1e-5),:] # non_zero dimension extract in estimator
    t = beta[nonzero_dim(beta,1e-5),:] # non_zero dimension of true beta
    
    
    
    r1 = torch.matrix_rank(est).item() # rank of estimator
    r2 = torch.matrix_rank(t).item() # rank of true beta
    
    # add zero vector in smaller rank parameter
    if r1 - r2 > 0:
        add_vec = torch.zeros(r1 -r2, t.size()[1])
        t = torch.cat((t,add_vec), dim = 0) # zero vector add in true beta 
    elif r1 - r2 < 0:
        add_vec = torch.zeros(r2 -r1, est.size()[1])
        est = torch.cat((est,add_vec), dim = 0) # zero vector add in estimator        
    else:
        pass

        
    if mode == 'predictor':
        est_x = est @ torch.t(X) # mode equals predictor then compute mcorr using beta^{T} X
        true_x = t @ torch.t(X)
    else:
        est_x = est # using only beta
        true_x = t
        
    n = est_x.size()[1]
    
    
    # minus mean of each values
    
    est_x -= torch.mean(est_x, dim=1).unsqueeze(1)
    true_x -= torch.mean(true_x, dim=1).unsqueeze(1)
    
    # calculate covariance matrix 
    
    c_uu = est_x @ torch.t(est_x) / n
    c_vv = true_x @ torch.t(true_x) / n

    c_vu = true_x @ torch.t(est_x) / n
    c_uv =  est_x @ torch.t(true_x)/ n
        
    c_vv_sqrt = pow_cov(c_vv, -0.5)
    c_uu_inv =  pow_cov(c_uu, -1)
    
    whole = c_vv_sqrt @ c_vu @ c_uu_inv @ c_uv @ c_vv_sqrt
    
    # multiple correlation between true predictor and estimate predictor
    
    measure_1 = torch.trace(whole) / torch.matrix_rank(whole)
    
    return measure_1.item()

def F_norm(W, beta):
    
    est = W[nonzero_dim(W, 1e-5),:]
    t = beta[nonzero_dim(beta,1e-5),:]
    
    r1 = torch.matrix_rank(est).item()
    r2 = torch.matrix_rank(t).item()
    
    if r1 - r2 > 0:
        add_vec = torch.zeros(r1 -r2, est.size()[1])
        t = torch.cat((t,add_vec), dim = 0) # zero vector add in true beta 
    elif r1 - r2 < 0:
        add_vec = torch.zeros(r2 -r1, est.size()[1])
        est = torch.cat((est,add_vec), dim = 0) # zero vector add in estimator
    else:
        pass
        
    est_mul = est @ torch.t(est)
    t_mul = t @ torch.t(t)
    
    
    P = est_mul - t_mul
    value = torch.linalg.norm(P, ord = "fro")
    return value.item()

# def dim_normalize(W):
#     # in terms of space, normalize of the matrix does not affect the spanning system
#     W_tr = torch.zeros(W.size()[0], W.size()[1])
#     norm_vec = torch.norm(W, dim = 0)
#     for i in range(0,len(norm_vec)):
#         if norm_vec[i] != 0:
#             W_tr[i,:] = W[i,:] / norm_vec[i]
#         else:
#             W_tr[i,:] = W[i,:]
            
#     return W_tr

# input matrix is given by p * d

def general_loss(M1, M2):
    
    # normalized in terms of column vector
    non_M1 = torch.nn.functional.normalize(M1, dim=0)
    non_M2 = torch.nn.functional.normalize(M2, dim=0)
    
    mat1 = torch.mm(non_M1, non_M1.T)
    mat2 = torch.mm(non_M2, non_M2.T)
    # loss between two linear subspace
    loss = torch.norm(mat1 - mat2)**2
    return loss.item()

def projection_loss(M1, M2):
    
    # normalized in terms of column vector
    non_M1 = torch.nn.functional.normalize(M1, dim=0)
    non_M2 = torch.nn.functional.normalize(M2, dim=0)
    
    mid1 = torch.mm(non_M1.T, non_M1)
    mid2 = torch.mm(non_M2.T, non_M2)
    
    inv1 = torch.inverse(mid1)
    inv2 = torch.inverse(mid2)
    
    last_1 = torch.mm(inv1, non_M1.T)
    last_2 = torch.mm(inv2, non_M2.T)
    
    pro1 = torch.mm(non_M1, last_1)
    pro2 = torch.mm(non_M2, last_2)

    # loss between projection matrix P_A, P_B
    loss = torch.norm(pro1 - pro2)**2
    return loss.item()


def Correlation_loss(M1, M2, X):
    
    # M1, M2 implies p * d matrix and X is n * p matrix
    # M1 is estimate and M2 is true subspace
    
    d = M1.size()[1]
    sig = torch.cov(X.T)
    
    non_M1 = torch.nn.functional.normalize(M1, dim=0)
    non_M2 = torch.nn.functional.normalize(M2, dim=0)
    
    first_term = torch.matmul(torch.matmul(non_M1.T, sig), non_M1)
    second_term = torch.matmul(torch.matmul(non_M1.T, sig), non_M2)
    third_term = torch.matmul(torch.matmul(non_M2.T, sig), non_M2)
    final_term  = torch.matmul(torch.matmul(non_M2.T, sig), non_M1)
    
    inv_1 = torch.inverse(first_term)
    inv_3 = torch.inverse(third_term)
    
    tr_term = torch.matmul(torch.matmul(inv_1, second_term) , torch.matmul(inv_3, final_term))
    sum_tr = torch.trace(tr_term) / d
    return 1.0 - sum_tr.item()

def CMS(model, X):
    
    num_n = X.size()[0]
    num_p = X.size()[1]
    half_features = int(num_p * 0.5)
    
    
    if model == 'model1':
        beta1 = torch.zeros(num_p, 1)
        beta1[0] = 1.0
        beta1[1] = 0.5
        beta1[2] = 1.0
        true_beta = beta1 / torch.norm(beta1)
    elif model == 'model2':
        beta1 = torch.zeros(num_p, 1)
        beta1[0] = 1.0
        beta1[1] = 1.0
        beta1[2] = 1.0
        beta1[3] = 1.0
        true_beta = beta1 / torch.norm(beta1)
    elif model == 'model3':
        beta1 = torch.zeros(num_p, 1)
        beta1[0] = 1.0
        true_beta = beta1
    elif model == 'model4':
        beta1 = torch.zeros(num_p, 1)
        beta1[0] = 1.0
        beta1[2] = 1.0
        true_beta = beta1 / torch.norm(beta1)
    elif model == 'model5' or model == 'model8':
        beta1 = torch.zeros(num_p, 1)
        beta2 = torch.zeros(num_p, 1)
        
        beta1[:half_features] = 1.0
        beta2[:half_features:2] = 1.0
        beta2[1:half_features:2] = -1.0
        
        beta1 = beta1 / torch.norm(beta1)
        beta2 = beta2 / torch.norm(beta2)
        
        true_beta = torch.cat((beta1, beta2), dim=1)
    elif model == 'model6' or model == 'model7':
        beta1 = torch.zeros(num_p, 1)
        beta2 = torch.zeros(num_p, 1)
        
        beta1[0] = 1.0
        beta2[1] = 1.0
        
        true_beta = torch.cat((beta1, beta2), dim=1)
    else:
        raise ValueError
        
    return true_beta
        
    
    
    

