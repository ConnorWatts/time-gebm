### utils.DATALOADER.py


# Adapted from https://github.com/kevin-w-li/deep-kexpfam/blob/master/Datasets.py


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
#import h5py as h5
from scipy.stats import truncnorm as tnorm
from scipy.linalg import expm
#from autograd import elementwise_grad

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from sklearn.cluster import KMeans, SpectralClustering
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torch.autograd import Variable
import torch.functional as F


import os
import shutil #https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive # to unzip
#from shutil import make_archive # to create zip for storage
import requests #for downloading zip file
from scipy import io #for loadmat, matlab conversion
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt # for plotting - pandas uses matplotlib
#from tabulate import tabulate # for verbose tables
#from tensorflow.keras.utils import to_categorical # for one-hot encoding

#credit https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
#many other methods I tried failed to download the file properly
from torch.utils.data import Dataset, DataLoader

#data augmentation
#import tsaug




def sine_data_generation (no, seq_len, dim):

  ## from https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py

  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0.02, 0.1)            
      phase = np.random.uniform(-3.14, 3.14)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)

    
                
  return data


class SineDataset(Dataset):
    def __init__(self,args,no, seq_len, dim):
        self.data = sine_data_generation (no, seq_len, dim)
        self.data = np.array(self.data)
        if args.generator == "tts":
            self.data = np.transpose(self.data, (0, 2, 1))
            self.data = np.expand_dims(self.data, axis=2)

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx],self.data[idx]

### from original timeGAN paper

def MinMaxScaler(data):
  
  ### from https://github.com/jsyoon0823/TimeGAN
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

#%% Load Google Data
    
def google_data_loading (seq_length):

    #### from https://github.com/jsyoon0823/TimeGAN
    # Load Google Data
    x = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
    # Flip the data to make chronological data
    x = x[:,:-1]
    x = x[::-1]
    # Min-Max Normalizer
    x = MinMaxScaler(x)
    
    # Build dataset
    dataX = []
    
    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
    
    return outputX

def chickenpox_data_loading (seq_length):

    #### from https://github.com/jsyoon0823/TimeGAN
    # Load Google Data
    x = np.loadtxt('hungary_chickenpox.csv', delimiter = ",",skiprows = 1,usecols=range(1,21))

    #x = x[:,1:]
    #x = x[::-1]
    # Min-Max Normalizer
    x = MinMaxScaler(x)
    
    # Build dataset
    dataX = []
    
    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
    
    return outputX

def energy_data_loading (seq_length):

    #### from https://github.com/jsyoon0823/TimeGANoriginal timeGAN paper
    # Load Google Data
    x = np.loadtxt('energy_data.csv', delimiter = ",",skiprows = 1)
    # Flip the data to make chronological data
    x = x[::-1]
    # Min-Max Normalizer
    x = MinMaxScaler(x)
    
    # Build dataset
    dataX = []
    
    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
    
    return outputX

def gaus_data_loading(seq_len,phi,sigma,no,features):
  mean = np.zeros(features)
  cov1 = sigma * np.ones((features,features))
  cov2 =  (1-sigma) * np.identity(features)
  cov = cov1 + cov2
  #covariance
  data = list()
  for i in range(no):
    sequence = list()
    x= np.random.multivariate_normal(mean, cov, seq_len)
    for j in range(seq_len):
      if j == 0:
        x_ = x[j]
      else:
        x_ = phi * x_ + x[j]
      sequence.append(x_)
    data.append(sequence)
  return data

class GausDataset(Dataset):
    def __init__(self,args,data):

        self.data = data
        self.data = np.array(self.data)
        if args.generator == "tts":
            self.data = np.transpose(self.data, (0, 2, 1))
            self.data = np.expand_dims(self.data, axis=2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx],self.data[idx]

class StockDataset(Dataset):
    def __init__(self,args,data):
        self.data = data
        self.data = np.array(self.data)
        if args.generator == "tts":
            self.data = np.transpose(self.data, (0, 2, 1))
            self.data = np.expand_dims(self.data, axis=2)


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx],self.data[idx]

class ChickenpoxDataset(Dataset):
    def __init__(self,args,data):
        self.data = data
        self.data = np.array(self.data)
        if args.generator == "tts":
            self.data = np.transpose(self.data, (0, 2, 1))
            self.data = np.expand_dims(self.data, axis=2)


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx],self.data[idx]

class EnergyDataset(Dataset):
    def __init__(self,args,data):
        self.data = data
        self.data = np.array(self.data)
        if args.generator == "tts":
            self.data = np.transpose(self.data, (0, 2, 1))
            self.data = np.expand_dims(self.data, axis=2)


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx],self.data[idx]


    
