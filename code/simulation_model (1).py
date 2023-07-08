import torch
from common import indicator
from math import pi


##### model1 to model


def model1(X):
    
    # define number of samples and features
    
    num_features = X.size()[1]
    num_size = X.size()[0]
    # half_var = int(0.5 * num_features)
    
    # define true beta0 which implies central mean subspace
    # true dimension is two
    beta1 = torch.zeros(num_features, 1) # true beta is B0 = (e1, e2)
    beta1[0] = 1.0
    beta1[1] = 0.5
    beta1[2] = 1.0
    
    beta1 = beta1 / torch.norm(beta1)
    
    d1 = torch.mm(X, beta1) 
    y =  torch.exp(d1) + 0.5 * torch.normal(mean = 0 , std = 1, size = (num_size, 1))
    return y

def model2(X):
    
    # define number of samples and features
    
    num_features = X.size()[1]
    num_size = X.size()[0]
    # half_var = int(0.5 * num_features)
    
    # define true beta0 which implies central mean subspace
    # true dimension is two
    beta1 = torch.zeros(num_features, 1) # true beta is B0 = (e1, e2)
    beta1[0] = 1.0
    beta1[1] = 1.0
    beta1[2] = 1.0
    beta1[3] = 1.0
    
    beta1 = beta1 / torch.norm(beta1)
    
    d1 = torch.mm(X, beta1)
    y = 0.5 * (d1)**2 + 0.5 * torch.normal(mean = 0 , std = 1, size = (num_size, 1))
    return y

def model3(X):
    
    # define number of samples and features
    
    num_features = X.size()[1]
    num_size = X.size()[0]

    beta1 = torch.zeros(num_features, 1) 
    beta1[0] = 1.0

    
    d1 = torch.mm(X, beta1)
    y = torch.arcsin(1 / (1 + torch.abs(0.5 + d1))) + 0.2 * torch.normal(mean = 0 , std = 1, size = (num_size, 1))
    return y

def model4(X):
    
    # define number of samples and features
    
    num_features = X.size()[1]
    num_size = X.size()[0]
    # half_var = int(0.5 * num_features)
    
    # define true beta0 which implies central mean subspace
    # true dimension is two
    beta1 = torch.zeros(num_features, 1) # true beta is B0 = (e1, e2)
    beta1[0] = 1.0
    beta1[2] = 1.0
    
    beta1 = beta1 / torch.norm(beta1)
    
    d1 = torch.mm(X, beta1)
    y = torch.sin(d1) +  0.5 * torch.normal(mean = 0 , std = 1, size = (num_size, 1))
    return y



def model5(X):
    num_features = X.size()[1]
    num_size = X.size()[0]
    half_features = int(num_features * 0.5)
    
     # true dimension is three
    beta1 = torch.zeros(num_features, 1)
    beta2 = torch.zeros(num_features, 1)
    
    beta1[:half_features] = 1.0
    
    beta2[:half_features:2] = 1.0
    beta2[1:half_features:2] = -1.0
    
    beta1 = beta1 / torch.norm(beta1)
    beta2 = beta2 / torch.norm(beta2)
    
    d1 = torch.mm(X, beta1)
    d2 = torch.mm(X, beta2)
    
    y = d1 / (0.5 + (d2 + 1.5)**2) + 0.5 * torch.normal(mean = 0 , std = 1, size = (num_size, 1))
    return y


def model6(X):
    num_features = X.size()[1]
    num_size = X.size()[0]
    
     # true dimension is three
    beta1 = torch.zeros(num_features, 1)
    beta2 = torch.zeros(num_features, 1)
    
    beta1[0] = 1.0
    beta2[1] = 1.0    

    d1 = torch.mm(X, beta1)
    d2 = torch.mm(X, beta2)
    
    y = torch.cos(2 * d1) - torch.cos(d2) +  0.2 * torch.normal(mean = 0 , std = 1, size = (num_size, 1))
    return y

def model7(X):
    num_features = X.size()[1]
    num_size = X.size()[0]
    
     # true dimension is three
    beta1 = torch.zeros(num_features, 1)
    beta2 = torch.zeros(num_features, 1)
    
    beta1[0] = 1.0
    beta2[1] = 1.0    

    d1 = torch.mm(X, beta1)
    d2 = torch.mm(X, beta2)
    
    y = torch.sin(1.4 * d1) +(d2 + 1)**2 +  0.4 * torch.normal(mean = 0 , std = 1, size = (num_size, 1))
    return y


def model8(X):
    num_features = X.size()[1]
    num_size = X.size()[0]
    half_features = int(num_features * 0.5)
    
     # true dimension is three
    beta1 = torch.zeros(num_features, 1)
    beta2 = torch.zeros(num_features, 1)
    
    beta1[:half_features] = 1.0
    
    beta2[:half_features:2] = 1.0
    beta2[1:half_features:2] = -1.0
    
    beta1 = beta1 / torch.norm(beta1)
    beta2 = beta2 / torch.norm(beta2)
    
    d1 = torch.mm(X, beta1)
    d2 = torch.mm(X, beta2)
    
    y = d1**2 + d2**2 + 0.5 * torch.normal(mean = 0 , std = 1, size = (num_size, 1))
    return y




# def model3(X):
    
#     num_features = X.size()[1]
#     num_size = X.size()[0]
#     sigma = 0.2
    
#     # true dimension is two
    
#     beta1 = torch.ones(num_features, 1)
#     beta2 = torch.zeros(num_features, 1)
#     beta1[4:] = 0.0
#     beta2[(num_features-5):] = 1.0
    
#     d1 = torch.mm(X, beta1)
#     d2 = torch.mm(X, beta2)
    
#     form1 = d1 + sigma *  torch.randn(num_size, 1) -1
#     form2 = d2 + sigma *  torch.randn(num_size, 1)
    
#     pre1 =  indicator(form1)
#     pre2 = indicator(form2)
#     y = pre1 + 2 * pre2
#     return y

# def model4(X):
    
#     num_features = X.size()[1]
#     num_size = X.size()[0]
    
#     beta1 = torch.ones(num_features, 1)
#     beta1[1] = 0.5
#     beta1[4:] = 0.0
    
#     d1 = torch.mm(X, beta1)
#     y = torch.exp(d1) * torch.randn(num_size, 1) * 0.2
#     return y

# def model5(X):
#     num_features = X.size()[1]
#     num_size = X.size()[0]
    
#     beta1 = torch.ones(num_features, 1)
#     beta1[5:] =  0.0
    
#     d1 = torch.mm(X, beta1)
#     y = 0.1 * (d1 + torch.randn(num_size, 1))**3
#     return y
    
# def model6(X):
#     num_features = X.size()[1]
#     num_size = X.size()[0]
    
#     beta1 = torch.zeros(num_features, 1)
#     beta2 = torch.zeros(num_features, 1)
#     beta1[0] = 1.0
#     beta2[1] = 1.0
    
#     d1 = torch.mm(X, beta1)
#     d2 = torch.mm(X, beta2)
    
#     y = torch.sign(2 * d1 + torch.randn(num_size, 1) ) * torch.log(torch.abs(2 * d2 + 4 + torch.randn(num_size, 1)))
#     return y
    
# #########################################################################################################################


# def model7(X):
#     num_features = X.size()[1]
#     num_size = X.size()[0]
    
#     beta1 = torch.ones(num_features, 1)
#     beta2 = torch.ones(num_features, 1)
#     vec_norm = torch.norm(beta1)
#     beta2[::2] = -1.0
    
#     beta1 = beta1 / torch.norm(beta1)
#     beta2 = beta1 / torch.norm(beta2)
    
#     d1 = torch.mm(X, beta1)
#     d2 = torch.mm(X, beta2)
    
#     y = d1**2 + 2* torch.abs(d2) + 0.1 * torch.abs(d2) * torch.randn(num_size, 1)
#     return y

# def model8(X):
#     num_features = X.size()[1]
#     num_size = X.size()[0]
    
#     beta1 = torch.ones(num_features, 1)
#     beta2 = torch.ones(num_features, 1)
#     vec_norm = torch.norm(beta1)
#     beta2[::2] = -1.0
    
#     beta1 = beta1 / torch.norm(beta1)
#     beta2 = beta1 / torch.norm(beta2)
    
#     d1 = torch.mm(X, beta1)
#     d2 = torch.mm(X, beta2)
#     y = torch.exp(d1) + 2 *(d2 + 1)**2 + torch.abs(d1)*torch.randn(num_size, 1)
#     return y

# def model9(X):
#     num_features = X.size()[1]
#     num_size = X.size()[0]
    
#     beta1 = torch.ones(num_features, 1)
#     beta2 = torch.ones(num_features, 1)
#     vec_norm = torch.norm(beta1)
#     beta2[::2] = -1.0
    
#     beta1 = beta1 / torch.norm(beta1)
#     beta2 = beta1 / torch.norm(beta2)
    
#     d1 = torch.mm(X, beta1)
#     d2 = torch.mm(X, beta2)
    
#     y = d1**2 + d2**2 + 0.5 * torch.randn(num_size, 1)
#     return y

