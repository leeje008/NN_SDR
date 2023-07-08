import torch
from measure import *
import torch.nn as nn


class BsplineActivation(torch.nn.Module):
    def __init__(self):
        super(BsplineActivation, self).__init__()
        return
    def forward(self, x):
        if x < -1:
            return 0
        elif x < 0 :
            return x + 1
        elif x < 1:
            return 1 - x
        else:
            return 0

def l2_loss(odict1, odict2):
    """
    Compute the L2 loss between two OrderedDicts.

    Args:
        odict1 (OrderedDict): The first OrderedDict.
        odict2 (OrderedDict): The second OrderedDict.

    Returns:
        The L2 loss between the two OrderedDicts as a PyTorch tensor.
    """
    # check if one of the values is None, and return the other object if true
    if None in [odict1, odict2]:
        return 0.0

    
    
    assert len(odict1) == len(odict2), "The two OrderedDicts must have the same length."
    assert list(odict1.keys()) == list(odict2.keys()), "The two OrderedDicts must have the same keys."

    
    squared_sum = 0.0
    for key in odict1.keys():
        mat1 = odict1[key]
        mat2 = odict2[key]
        assert mat1.shape == mat2.shape, f"The matrices for key {key} must have the same shape."
        mat_diff = mat1 - mat2
        squared_diff = torch.sum(mat_diff**2)
        squared_sum += squared_diff
        
    return squared_sum.item()

def GroupLasso_update(X, lam1, lr):
    target = X
        
    m = torch.nn.Threshold(0.0, 0.0)
        
    for i in range(target.size()[0]):
        l2_norm = torch.norm(target[i,:], p =2)
        input_val = 1 -(lr * lam1) / l2_norm
        output_val = m(input_val)
        target[i,:] = output_val * target[i,:]
        
    return target

def GroupLasso_update_p(X, lam1, lr):
    target = X
        
    m = torch.nn.Threshold(0.0, 0.0)
        
    for i in range(target.size()[1]):
        l2_norm = torch.norm(target[:,i], p =2)
        input_val = 1 -(lr * lam1) / l2_norm
        output_val = m(input_val)
        target[:,i] = output_val * target[:,i]
        
    return target

def p_adj_pen(lam1, parameter):
    norm = torch.tensor([torch.norm(parameter[:,i]) for i in range(parameter.size()[1])])
    pen_value = lam1 * norm
    return torch.sum(pen_value)

def ortho_penalty(w1, sig ,lam0):
    norm = 0.0
    mul1 = torch.mm(w1, sig)
    mul2 = torch.mm(mul1, w1.T)
    differ = mul2 - torch.eye(w1.shape[0])
    f_norm = torch.norm(differ, p='fro')**2
    norm += lam0 * f_norm
    return norm

def pro_penalty(w1, lam0):
    norm = 0.0
    mul = torch.mm(w1, w1.T)
    differ = mul - torch.eye(w1.shape[0])
    f_norm = torch.norm(differ, p='fro')**2
    norm += lam0 * f_norm
    return norm

def none_or_path(value):
    if value is None:
        return None
    return str(value)


def sort_by_indices(tensor, indices):
    # 인덱스를 기준으로 정렬
    sorted_tensor = torch.index_select(tensor, 0, indices.sort()[1])

    return sorted_tensor


def adj_pen(lam1, parameter):
    norm = torch.tensor([torch.norm(parameter[i,:]) for i in range(parameter.size()[0])])
    pen_value = lam1 * norm
    return torch.sum(pen_value)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)   



class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))
        
        
def indicator(x):
    return torch.where(x > 0, torch.tensor(1.), torch.tensor(0.))