import torch
import numpy as np
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import activate
import rpy2.robjects as robjects


def torch_tensor_to_r_matrix(x_torch):
    numpy2ri.activate()
    r = importr('base')
    x_np = x_torch.detach().cpu().numpy()
    return r.matrix(numpy2ri.numpy2rpy(x_np), nrow=x_np.shape[0], ncol=x_np.shape[1])


def torch_tensor_to_r_vector(x_torch):
    numpy2ri.activate()
    r = importr('base')
    x_np = x_torch.detach().cpu().numpy()
    r_vector = r.vector(numpy2ri.numpy2rpy(x_np))
    return r_vector


def sdr_init(X, y, dim):
    X_r = torch_tensor_to_r_matrix(X)
    y_r = torch_tensor_to_r_matrix(y)
    
    # activate R script
    robjects.r['source']('dr_init.R')
    # define r object
    r_func = robjects.r['dr']
    
    # assume regression problem
    result = r_func(X_r, y_r, h = 5, r = dim, ytype="continuous")
    
    py_dict = dict(zip(result.names, list(result)))
    for k, v in py_dict.items():
        py_dict[k] = torch.tensor(np.array(v), dtype=torch.float32)
        
    return py_dict['beta']


def sdr(X, y, dim, mode = 'sir'):

    X_r = torch_tensor_to_r_matrix(X)
    y_r = torch_tensor_to_r_matrix(y)
    
    # activate R script
    robjects.r['source']('sdr.R')
    
    # define sdr function
    
    if mode == 'sir':
        r_func = robjects.r['sir']
        result = r_func(X_r, y_r, h = 5, r = dim, ytype="continuous")
    elif mode == 'save':
        r_func = robjects.r['save']
        result = r_func(X_r, y_r, h = 5, r = dim, ytype="continuous")
    elif mode == 'dr':
        r_func = robjects.r['dr']
        result = r_func(X_r, y_r, h = 5, r = dim, ytype="continuous")
    elif mode == 'mave':
        r_func = robjects.r['mave']
        result = r_func(X_r, y_r, d = dim, nit = 10)
    elif mode == 'rmave':
        r_func = robjects.r['rmave']
        result = r_func(X_r, y_r,d = dim, nit = 10)
    elif mode == 'opg':
        r_func = robjects.r['opg']
        result = r_func(X_r, y_r,  d = dim)
    else:
        raise Valuerror

    res_to = torch.tensor(np.array(result), dtype=torch.float32)

    return res_to