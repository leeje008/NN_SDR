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

def sdr(X, y, dim, mode = 'sir'):
    X_r = torch_tensor_to_r_matrix(X)
    y_r = torch_tensor_to_r_matrix(y)
    
    # activate R script
    robjects.r['source']('sdr.R')
    # define r object
    r_func = robjects.r['sir']
    
    # assume regression problem
    result = r_func(X_r, y_r, h = 5, r = dim, ytype="continuous")
    
    py_dict = dict(zip(result.names, list(result)))
    for k, v in py_dict.items():
        py_dict[k] = torch.tensor(np.array(v))
        
    return py_dict['beta'], py_dict['adj_pen'], py_dict['eigen_val']