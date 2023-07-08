import torch
from torch.utils.data import Dataset
from simulation_model import *
from stat_sampler import *

# generate Simulation dataset using number of variable, number of samples and model equation (need to be self defined)
class SyntheticDataset(Dataset):
    def __init__(self, sample_size, num_features, model_num , seed=None):
        super().__init__()
        self.sample_size = sample_size
        self.num_features = num_features
        self.model_num = model_num
        self.seed = seed
        
        if seed is not None:
            torch.manual_seed(seed)
        
        mul_sample = sample_from_multivariate(num_samples = self.sample_size, num_variables = self.num_features, seed = self.seed)
        
        # Generate random input values
        self.X = mul_sample.sampler()
        
        # Generate output values based on the model function
        with torch.no_grad():
            if model_num == 'model1':
                self.y = model1(self.X) 
            elif model_num == 'model2':
                self.y = model2(self.X) 
            elif model_num == 'model3':
                self.y = model3(self.X) 
            elif model_num == 'model4':
                self.y = model4(self.X) 
            elif model_num == 'model5':
                self.y = model5(self.X) 
            elif model_num == 'model6':
                self.y = model6(self.X) 
            elif model_num == 'model7':
                self.y = model7(self.X) 
            elif model_num == 'model8':
                self.y = model8(self.X) 
            elif model_num == 'model9':
                self.y = model9(self.X)
            else:
                raise print("Not accetable")
    
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

# Load data set using assume csv

class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        x = torch.tensor(sample[:-1].values, dtype=torch.float32)
        y = torch.tensor(sample[-1], dtype=torch.float32)
        return x, y