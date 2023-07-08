import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import warnings
warnings.filterwarnings('ignore')

class sample_from_multivariate:
    def __init__(self, num_samples, num_variables, seed):
        self.num = num_samples
        self.var = num_variables
        self.seed = seed 
    def p_cov(self):
        cov_mat = torch.eye(self.var)
        for i in range(0,cov_mat.size()[0]):
            for j in range(0,cov_mat.size()[1]):
                if i != j:
                    cov_mat[i,j] = 0.5**(abs(i-j))
        return cov_mat
    def sampler(self):
        torch.manual_seed(self.seed)
        m = MultivariateNormal(torch.zeros(self.var), self.p_cov())
        return m.sample_n(n = self.num)
