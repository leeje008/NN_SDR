import torch
import torch.nn as nn
from common import *

# main model structure

class ray_SDR_net(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim):
        super(ray_SDR_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = h1_dim
        self.hidden_dim2 = h2_dim
        
        # in this case solve regression problem
        
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.layer2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.layer3 = nn.Linear(self.hidden_dim2, self.hidden_dim2)
        self.layer4 = nn.Linear(self.hidden_dim2, 1)

    def forward(self, X):
        
        # activation function
        
        out_h1 = torch.tanh(self.layer1(X))
        out_h2 = torch.relu(self.layer2(out_h1))
        out_h3 = torch.tanh(self.layer3(out_h2))
        final_out = self.layer4(out_h3)
        
        return final_out
    
    def custom_weight_init(self, weight_dict, seed = None):
        if seed is not None:
            torch.manual_seed(seed)
        for name, param in self.named_parameters():
            # using custom weight init
            if 'weight' in name:
                if name in weight_dict:
                    param.data = weight_dict[name].data
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
            else :
                # Use random initialization method using xaiver
                torch.nn.init.xavier_normal_(param)
                
class bn_SDR_net(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim):
        super(bn_SDR_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = h1_dim
        self.hidden_dim2 = h2_dim

        
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.layer2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.layer3 = nn.Linear(self.hidden_dim2, 1)
        
        self.bn1 = nn.BatchNorm1d(self.hidden_dim1)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim2)

    def forward(self, X):
        out_dim = self.layer1(X.float())
        # out_h1 = torch.tanh(self.bn1(out_dim))
        out_h1 = torch.relu(self.bn1(out_dim))
        # out_h2 = self.layer2(out_h1)
        out_h2 = torch.sigmoid(self.bn2(self.layer2(out_h1)))
        final_out = self.layer3(out_h2)
        
        
        return final_out, out_dim
    
    def custom_weight_init(self, weight_dict, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for name, param in self.named_parameters():
            # using custom weight init
            if 'weight' in name:
                if name in weight_dict:
                    param.data = weight_dict[name].data
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
            else :
                torch.nn.init.xavier_normal_(param)
                
