### SAMPLERs.py

import torch
import numpy as np
import time
import math
from torch.autograd import Variable
from torch.autograd.variable import Variable
from torch import nn

import torch.nn.functional as F
#import compute as cp
#taken and modified from https://github.com/MichaelArbel/GeneralizedEBM/blob/master/samplers.py

 
class Latent_potential(nn.Module):

    def __init__(self, generator,discriminator,latent_prior, temperature=100):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_prior = latent_prior
        self.temperature = temperature
    def forward(self,Z):
        if Z.size(dim=1) == 32:
            with torch.backends.cudnn.flags(enabled=False):
                g_states = self.generator.init_hidden(Z.shape[0])
                d_state = self.discriminator.init_hidden(Z.shape[0])
                out,_ = self.generator(Z,g_states)
                out,_,_ =  self.discriminator(out,d_state)
              #should it be mean!!
                out = -self.latent_prior.log_prob(Z).mean(dim=1) + self.temperature*out
        else:
            out = self.generator(Z)
            out =  self.discriminator(out)
            out = -self.latent_prior.log_prob(Z) + self.temperature*out

        ## old
        #out = self.generator(Z)
        #out =  self.discriminator(out)
        #out = -self.latent_prior.log_prob(Z) + self.temperature*out 

        return out

class Cold_Latent_potential(nn.Module):

    def __init__(self, generator,discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
    def forward(self,Z):
        out = self.generator(Z)
        out =  self.discriminator(out)        
        return out

class Independent_Latent_potential(nn.Module):

    def __init__(self, generator,discriminator,latent_prior):
        #super(Latent_potential).__init__()
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_prior = latent_prior
    def forward(self,Z):
        #with torch.no_grad():
        out = self.generator(Z)
        out =  self.discriminator(out)
        return out 

class Dot_Latent_potential(nn.Module):

    def __init__(self, generator,discriminator,latent_prior):
        #super(Latent_potential).__init__()
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_prior = latent_prior
    def forward(self,Z):
        #with torch.no_grad():
        out = self.generator(Z)
        out =  self.discriminator(out)
        return  torch.norm(Z, dim=1) + out 

class Grad_potential(nn.Module):
    def __init__(self, potential):
        super().__init__()
        self.potential = potential
    def forward(self,X):
        X.requires_grad_()
        out = self.potential(X).sum()
        out.backward()
        return X.grad

class Grad_cond_potential(nn.Module):
    def __init__(self, potential):
        super().__init__()
        self.potential = potential
    def forward(self,X, labels):
        X.requires_grad_()
        Z = X,labels
        out = self.potential(Z).sum()
        out.backward()
        return X.grad


class LangevinSampler(object):
    def __init__(self, potential, T=100,  gamma=1e-2):          
        
        self.potential = potential
        
        #self.num_steps_min = num_steps_min
        #self.num_steps_max = num_steps_max
        self.gamma = gamma
        self.grad_potential = Grad_potential(self.potential)
        self.T = T
        #self.grad_momentum = Grad_potential(self.momentum.log_prob)
        #self.sampler_momentum = momentum_sampler 
        
    def sample(self,prior_z,sample_chain=False,T=None,thinning=10):
        if T is None:
            T = self.T
        sampler = torch.distributions.Normal(torch.zeros_like(prior_z), 1.)
        
        #self.momentum.eval()

        #old
        self.potential.eval()

        #self.potential.train()

        t_extract_list = []
        Z_extract_list = []
        accept_list = []

        Z_t = prior_z.clone().detach()
        
        gamma = 1.*self.gamma
        #print(f'Initial lr: {gamma}')
        for t in range(T):
            if sample_chain and t > 0 and t % thinning == 0:
                t_extract_list.append(t)
                Z_extract_list.append(Z_t.clone().detach().cpu())
                accept_list.append(1.)

            # reset computation graph
            Z_t = self.euler(Z_t,self.grad_potential,sampler,gamma=gamma)
            # only if extracting the samples so we have a sequence of samples
            if t>0 and t%200==0:
                gamma *=0.1
                print('decreasing lr for sampling')

            #print('iteration: '+ str(t))
        if not sample_chain:
            return Z_t.clone().detach(),1.
        return t_extract_list, Z_extract_list, accept_list


    def euler(self,x,grad_x,sampler,gamma=1e-2):
        x_t = x.clone().detach()
        D = np.sqrt(gamma)
        x_t = x_t - gamma / 2 * grad_x(x_t) + D * sampler.sample()
        return x_t

class ZeroTemperatureSampler(object):
    def __init__(self, potential, T=100,  gamma=1e-2):          
        
        self.potential = potential
        
        #self.num_steps_min = num_steps_min
        #self.num_steps_max = num_steps_max
        self.gamma = gamma
        self.grad_potential = Grad_potential(self.potential)
        self.T = T
        #self.grad_momentum = Grad_potential(self.momentum.log_prob)
        #self.sampler_momentum = momentum_sampler 
        
    def sample(self,prior_z,sample_chain=False,T=None,thinning=10):
        if T is None:
            T = self.T
        sampler = torch.distributions.Normal(torch.zeros_like(prior_z), 1.)
        
        #self.momentum.eval()
        self.potential.eval()
        t_extract_list = []
        Z_extract_list = []
        accept_list = []

        Z_t = prior_z.clone().detach()
        
        gamma = 1.*self.gamma
        #print(f'Initial lr: {gamma}')
        for t in range(T):
            if sample_chain and t > 0 and t % thinning == 0:
                t_extract_list.append(t)
                Z_extract_list.append(Z_t.clone().detach().cpu())
                accept_list.append(1.)

            # reset computation graph
            Z_t = self.euler(Z_t,self.grad_potential,sampler,gamma=gamma)
            # only if extracting the samples so we have a sequence of samples
            if t>0 and t%200==0:
                gamma *=0.1
                print('decreasing lr for sampling')

            #print('iteration: '+ str(t))
        if not sample_chain:
            return Z_t.clone().detach(),1.
        return t_extract_list, Z_extract_list, accept_list


    def euler(self,x,grad_x,sampler,gamma=1e-2):
        x_t = x.clone().detach()
        D = np.sqrt(gamma)
        x_t = x_t - gamma / 2 * grad_x(x_t) #+ D * sampler.sample()
        return x_t

class SphereLangevinSampler(object):
    def __init__(self, potential, T=100,  gamma=1e-2):          
        
        self.potential = potential
        self.gamma = gamma
        self.grad_potential = Grad_potential(self.potential)
        self.T = T
        
    def sample(self,prior_z,sample_chain=False,T=None,thinning=10):
        if T is None:
            T = self.T
        sampler = torch.distributions.Normal(torch.zeros_like(prior_z), 1.)
        
        #self.momentum.eval()
        self.potential.eval()
        t_extract_list = []
        Z_extract_list = []
        accept_list = []

        Z_t = prior_z.clone().detach()
        
        gamma = 1.*self.gamma
        #print(f'Initial lr: {gamma}')
        for t in range(T):
            if sample_chain and t > 0 and t % thinning == 0:
                t_extract_list.append(t)
                Z_extract_list.append(Z_t.clone().detach().cpu())
                accept_list.append(1.)

            # reset computation graph
            Z_t = self.euler(Z_t,self.grad_potential,sampler,gamma=gamma)
            # only if extracting the samples so we have a sequence of samples
            if t>0 and t%200==0:
                gamma *=0.1
                print('decreasing lr for sampling')

            #print('iteration: '+ str(t))
        if not sample_chain:
            return Z_t.clone().detach(),1.
        return t_extract_list, Z_extract_list, accept_list


    def euler(self,x,grad_x,sampler,gamma=1e-2):
        x_t = x.clone().detach()
        D = np.sqrt(2.*gamma)
        R = x_t.shape[1]
        grad = gamma * grad_x(x_t) 
        dot = torch.sum(grad*x_t, dim=1)
        #grad = grad -   torch.einsum('n,nd->nd',dot,x_t)/np.sqrt(R)
        x_t = x_t - grad + D * sampler.sample()
        return x_t






