### HELPERS.py


import argparse
import copy
import hashlib
import json
import numpy as np
import os
import time

import ast
#from torchvision.datasets import CIFAR10,ImageNet,DatasetFolder,LSUN
import torchvision.transforms as transforms

#from utils.celebA import CelebA

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#from models.generator import Generator
#from models.discriminator import Discriminator
#from models import energy_model

#import compute as cp

import time
from PIL import Image, ImageFilter
from utils.dataloader import SineDataset, google_data_loading, StockDataset, chickenpox_data_loading, ChickenpoxDataset, energy_data_loadinng, EnergyDataset
#import samplers
import sys


def get_sine_data_loader (args, b_size, num_workers_):
    seq_len = args.seq_length
    features = args.features
    train_set = SineDataset(args,10000,seq_len,features)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    test_set = SineDataset(args,2900,seq_len,features)
    test_loader = test_loader = torch.utils.data.DataLoader(test_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    val_set =  SineDataset(args,2900,seq_len,features)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    return train_loader,test_loader,val_loader, None

def get_stock_data_loader(args, b_size, num_workers_):
    seq_len = args.seq_length
    stock_data = google_data_loading (seq_len)
    stock_data_set = StockDataset(args,stock_data)
    train_loader = torch.utils.data.DataLoader(stock_data_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    test_loader = torch.utils.data.DataLoader(stock_data_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    val_loader = torch.utils.data.DataLoader(stock_data_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    return train_loader,test_loader,val_loader, None

def get_chickenpox_data_loader(args, b_size, num_workers_):
    seq_len = args.seq_length
    chickenpox_data = chickenpox_data_loading (seq_len)
    chickenpox_data_set = ChickenpoxDataset(args,chickenpox_data)
    train_loader = torch.utils.data.DataLoader(chickenpox_data_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    test_loader = torch.utils.data.DataLoader(chickenpox_data_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    val_loader = torch.utils.data.DataLoader(chickenpox_data_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    return train_loader,test_loader,val_loader, None

def get_energy_data_loader(args, b_size, num_workers_):
    seq_len = args.seq_length
    energy_data = energy_data_loading(seq_len)
    energy_data_set = EnergyDataset(args,energy_data)
    train_loader = torch.utils.data.DataLoader(energy_data_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    test_loader = torch.utils.data.DataLoader(energy_data_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    val_loader = torch.utils.data.DataLoader(energy_data_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    return train_loader,test_loader,val_loader, None

def get_gaus_data_loader(args, b_size, num_workers_):
    seq_len = args.seq_length
    phi = args.gaus_phi
    sigma = args.gaus_sigma
    no = 3000
    features = args.features
    gaus_data = gaus_data_loading(seq_len,phi,sigma,no,features)
    gaus_data_set = GausDataset(args,gaus_data)
    train_loader = torch.utils.data.DataLoader(gaus_data_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    test_loader = torch.utils.data.DataLoader(gaus_data_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    val_loader = torch.utils.data.DataLoader(gaus_data_set, batch_size=b_size, num_workers=num_workers_, shuffle = True)
    return train_loader,test_loader,val_loader, None


def get_data_loader(args, b_size,num_workers):
    if  args.dataset_type == 'Sine':
        trainloader,testloader,validloader, input_dims = get_sine_data_loader(args, b_size,num_workers) 
    elif args.dataset_type == 'Stock':
        trainloader,testloader,validloader, input_dims = get_stock_data_loader(args, b_size,num_workers) 
    elif args.dataset_type == 'Energy':
        trainloader,testloader,validloader, input_dims = get_energy_data_loader(args, b_size,num_workers)
    elif args.dataset_type == 'Chickenpox':
        trainloader,testloader,validloader, input_dims = get_chickenpox_data_loader(args, b_size,num_workers)
    elif args.dataset_type == 'Gaus':
        trainloader,testloader,validloader, input_dims = get_gaus_data_loader(args, b_size,num_workers)    
    return trainloader,testloader,validloader, input_dims


# choose loss type
def get_loss(args):
    if args.criterion=='hinge':
        return hinge
    elif args.criterion=='wasserstein':
        return wasserstein
    elif args.criterion=='logistic':
        return logistic
    elif args.criterion == 'lsgan':
        return lsgan
    elif args.criterion == 'rnn_loss':
        return rnn_loss
    elif args.criterion in ['kale', 'donsker']:
        return kale
    elif args.criterion=='kale-nlp':
        return kale

# choose the optimizer
def get_optimizer(args, net_type, params):
    if net_type == 'discriminator':
        learn_rate = args.lr
    elif net_type == 'generator':
        learn_rate = args.lr_generator
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=learn_rate, weight_decay=args.weight_decay, betas = (args.beta_1,args.beta_2))
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=learn_rate, momentum=args.sgd_momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError('optimizer {} not implemented'.format(args.optimizer))
    return optimizer

# schedule the learning
def get_scheduler(args,optimizer):
    if args.scheduler=='MultiStepLR':
        if args.milestone is None:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.total_epochs*0.5), int(args.total_epochs*0.75)], gamma=args.lr_decay)
        else:
            milestone = [int(_) for _ in args.milestone.split(',')]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=args.lr_decay)
        return lr_scheduler
    elif args.scheduler=='ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)


def get_normal(Z_dim, device):
    loc = torch.zeros(Z_dim).to(device)
    scale = torch.ones(Z_dim).to(device)
    normal = torch.distributions.normal.Normal(loc, scale)
    return torch.distributions.independent.Independent(normal,1)

class ConditionalNoiseGen(nn.Module):
    def __init__(self, truncation,device):
        super(ConditionalNoiseGen, self).__init__()
        self.truncation = truncation
        self.device = device
        self.num_classes = 1000
        labels = 1.*np.array(range(self.num_classes))/self.num_classes
        labels= torch.tensor(list(labels)).to(device)
        self.multinomial = torch.distributions.categorical.Categorical(labels)
    def log_prob(self,noise):
        Z,labels = noise
        prob = - 0.5 * torch.norm(Z, dim=1) ** 2
        return prob

    def sample(self,shape):
        Z = truncated_noise_sample(truncation=self.truncation, batch_size=shape[0])
        Z = torch.from_numpy(Z)
        Z = Z.to(self.device)
        labels = self.multinomial.sample(shape)
        return Z,labels

# return the distribution of the latent noise

def get_latent_sampler(args,potential,Z_dim,device):
    momentum = get_normal(Z_dim,device)
    if args.latent_sampler=='hmc':
        return HMCsampler(potential,momentum, T=args.num_sampler_steps, num_steps_min=10, num_steps_max=20,gamma=args.lmc_gamma,  kappa = args.lmc_kappa)
    elif args.latent_sampler=='lmc':
        return LMCsampler(potential,momentum, T=args.num_sampler_steps, num_steps_min=10, num_steps_max=20,gamma=args.lmc_gamma,  kappa = args.lmc_kappa)
    elif args.latent_sampler=='langevin':
        return LangevinSampler(potential,  T=args.num_sampler_steps,gamma=args.lmc_gamma)
    elif args.latent_sampler=='zero_temperature_langevin':
        return ZeroTemperatureSampler(potential,  T=args.num_sampler_steps,gamma=args.lmc_gamma)
    elif args.latent_sampler=='mala':
        return MALA(potential,  T=args.num_sampler_steps,gamma=args.lmc_gamma)
    elif args.latent_sampler=='spherelangevin':
        return SphereLangevinSampler(potential,  T=args.num_sampler_steps,gamma=args.lmc_gamma)
    elif args.latent_sampler=='dot':
        return DOT(potential,  T=args.num_sampler_steps,gamma=args.lmc_gamma)
    elif args.latent_sampler=='trunclangevin':
        return TruncLangevinSampler(potential,momentum,trunc=args.trunc,  T=args.num_sampler_steps, num_steps_min=10, num_steps_max=20,gamma=args.lmc_gamma,  kappa = args.lmc_kappa)
    elif args.latent_sampler=='mh':
        return MetropolisHastings(potential, T=args.num_sampler_steps, gamma=args.lmc_gamma )
    elif args.latent_sampler=='imh':
        return IndependentMetropolisHastings(potential,  T=args.num_sampler_steps, gamma=args.lmc_gamma)


def get_latent_noise(args,dim,device):
    #dim = int(dim)
    if args.latent_noise=='gaussian':
        loc = torch.zeros(dim).to(device)
        scale = torch.ones(dim).to(device)
        normal = torch.distributions.normal.Normal(loc, scale)
        return torch.distributions.independent.Independent(normal,1)
    elif args.latent_noise=='uniform':
        return torch.distributions.Uniform(torch.zeros(dim).to(device),torch.ones(dim).to(device))


# return a discriminator for the energy model
def get_energy(args,input_dims,device):
    if args.discriminator =='crnn':
        features = args.features
        return DiscriminatorCRNN(num_feats = features, use_cuda=True ).to(device)
    elif args.discriminator=='tts':
        features = args.features
        ts_length = args.seq_length
        return DiscriminatorTTS(in_channels = features, seq_length = ts_length).to(device)


# return the base for the energy model
def get_base(args,input_dims,device):
    if args.generator =='crnn':
        features = args.features
        return GeneratorCRNN(num_feats = features, use_cuda=True ).to(device)
    elif args.generator=='tts':
        features = args.features
        ts_length = args.seq_length
        return GeneratorTTS(channels = features, seq_len = ts_length).to(device)


def init_logs(args, run_id, log_dir):
    if args.save_nothing:
        return None, None, None
    os.makedirs(log_dir, exist_ok=True)

    samples_dir = os.path.join(log_dir,  f'samples_{run_id}_{args.slurm_id}')
    os.makedirs(samples_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(log_dir, f'checkpoints_{run_id}_{args.slurm_id}')
    os.makedirs(checkpoint_dir, exist_ok=True)
                
    if args.log_to_file:
        log_file = open(os.path.join(log_dir, f'log_{run_id}_{args.slurm_id}.txt'), 'w', buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file        
    
    # log the parameters used in this run
    with open(os.path.join(log_dir, f'params_{run_id}_{args.slurm_id}.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)
    return log_dir,checkpoint_dir,samples_dir

def assign_device(device):
    if device >-1:
        device = 'cuda:'+str(device) if torch.cuda.is_available() and device>-1 else 'cpu'
    elif device==-1:
        device = 'cuda'
    elif device==-2:
        device = 'cpu'
    return device

def load_dictionary(file_name):
    out_dict = {}
    with open(file_name) as f:
        for line in f:
            cur_dict = json.loads(line)
            keys = cur_dict.keys()
            for key in keys:
                if key in out_dict:
                    out_dict[key].append(cur_dict[key])
                else:
                    out_dict[key] = [cur_dict[key]]
    return out_dict
