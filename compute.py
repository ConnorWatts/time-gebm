### COMPUTE.py

import os
import time

import numpy as np


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# fid_pytorch, inception
#import metrics.fid_pytorch as fid_pytorch
#from metrics.inception import InceptionV3
import torch
from scipy import linalg 


def rnn_loss(true_data,fake_data,loss_type):
  ## https://github.com/cjbayron/c-rnn-gan.pytorch/blob/master/train.py
  EPSILON = 1e-40 
  if loss_type=='discriminator':
    logits_real = torch.clamp(true_data, EPSILON, 1.0)
    d_loss_real = -torch.log(logits_real)

    if True:
      p_fake = torch.clamp((1 - logits_real), EPSILON, 1.0)
      d_loss_fake = -torch.log(p_fake)
      d_loss_real = 0.9*d_loss_real + 0.1*d_loss_fake

      logits_gen = torch.clamp((1 - fake_data), EPSILON, 1.0)
      d_loss_gen = -torch.log(logits_gen)

      batch_loss = d_loss_real + d_loss_gen
    return torch.mean(batch_loss)
  else:
    logits_gen = torch.clamp(fake_data, EPSILON, 1.0)
    batch_loss = -torch.log(logits_gen)

    return torch.mean(batch_loss)



def lsgan(true_data,fake_data,loss_type):
  if loss_type=='discriminator':
    if isinstance(fake_data, list):
      d_loss = 0
      for real_validity_item, fake_validity_item in zip(true_data, fake_data):
        real_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 1., dtype=torch.float, device=true_data.get_device())
        fake_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 0., dtype=torch.float, device=true_data.get_device())
        d_real_loss = nn.MSELoss()(real_validity_item, real_label)
        d_fake_loss = nn.MSELoss()(fake_validity_item, fake_label)
        d_loss += d_real_loss + d_fake_loss
    else:
      real_label = torch.full((true_data.shape[0],true_data.shape[1]), 1., dtype=torch.float, device=true_data.get_device())
      fake_label = torch.full((true_data.shape[0],true_data.shape[1]), 0., dtype=torch.float, device=true_data.get_device())
      d_real_loss = nn.MSELoss()(true_data, real_label)
      d_fake_loss = nn.MSELoss()(fake_data, fake_label)
      d_loss = d_real_loss + d_fake_loss
    return d_loss
  else:
    if isinstance(fake_data, list):
      g_loss = 0
      for fake_validity_item in fake_data:
        real_label = torch.full((fake_validity_item.shape[0],fake_validity_item.shape[1]), 1., dtype=torch.float, device=true_data.get_device())
        g_loss += nn.MSELoss()(fake_validity_item, real_label)
    else:
      real_label = torch.full((fake_data.shape[0],fake_data.shape[1]), 1., dtype=torch.float, device=true_data.get_device())
                        # fake_validity = nn.Sigmoid()(fake_validity.view(-1))
      g_loss = nn.MSELoss()(fake_data, real_label)
    return g_loss




def wasserstein(true_data,fake_data,loss_type):
    if loss_type=='discriminator':
        return -true_data.mean() + fake_data.mean()
    else:
        return -fake_data.mean()
def logistic(true_data,fake_data,loss_type):
    if loss_type =='discriminator':
        loss = torch.nn.BCEWithLogitsLoss()(true_data, torch.ones(true_data.shape[0]).to(true_data.device)) + \
                    torch.nn.BCEWithLogitsLoss()(fake_data, torch.zeros(fake_data.shape[0]).to(fake_data.device))
        return loss
    else:
        loss = torch.nn.BCEWithLogitsLoss()(fake_data, torch.ones(fake_data.shape[0]).to(fake_data.device))
        return loss
def kale(true_data,fake_data,loss_type):
    if loss_type=='discriminator':
        return  true_data.mean() + torch.exp(-fake_data).mean()  - 1
    else:
        return -true_data.mean() #- torch.exp(-fake_data).mean()  + 1


# calculates regularization penalty term for learning
def penalty_d(args, d, true_data, fake_data, device):
    penalty = 0.
    len_params = 0.
    # no penalty
    if args.penalty_type == 'none':
        pass
    # L2 regularization only
    elif args.penalty_type=='l2':
        for params in d.parameters():
            penalty += torch.sum(params**2)
    # gradient penalty only
    elif args.penalty_type=='gradient':
        penalty = _gradient_penalty(args,d, true_data, fake_data, device)
    # L2 + gradient penalty
    elif args.penalty_type=='gradient_l2':
        for params in d.parameters():
            penalty += torch.sum(params**2)
            len_params += np.sum(np.array(list(params.shape)))
        penalty = penalty/len_params
        g_penalty = _gradient_penalty(args,d, true_data, fake_data, device)
        penalty += g_penalty
    return penalty

# helper function to calculate gradient penalty
# adapted from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
def _gradient_penalty(args,d, true_data, fake_data, device):
    batch_size = true_data.size()[0]
    size_inter = min(batch_size,fake_data.size()[0])
    # Calculate interpolation
    shape  = list(np.ones(len(true_data.shape)-1))
    shape = tuple([int(a) for a in shape])
    alpha = torch.rand((size_inter,)+shape)
    alpha = alpha.expand_as(true_data)
    alpha = alpha.to(device)

    interpolated = alpha*true_data.data[:size_inter] + (1-alpha)*fake_data.data[:size_inter]
    #interpolated = torch.cat([true_data.data,fake_data.data],dim=0)
    interpolated = Variable(interpolated, requires_grad=True).to(device)

    # Calculate probability of interpolated examples
    with torch.backends.cudnn.flags(enabled=False):
      if args.discriminator == "crnn":
          d_state = d.init_hidden(true_data.shape[0])
          prob_interpolated,_,_ = d(interpolated,d_state)
      else:
          prob_interpolated = d(interpolated)

    # Calculate gradients of probabilities with respect to examples
      gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch

    ## important change
    #gradients = gradients.view(batch_size, -1)
    gradients = gradients.reshape(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sum(gradients ** 2, dim=1).mean()
    return gradients_norm




def iterative_mean(batch_tensor, total_mean, total_els, dim=0 ):
    b = batch_tensor.shape[dim]
    cur_mean = batch_tensor.mean(dim=dim)
    total_mean = (total_els/(total_els+b))*total_mean + (b/(total_els+b))*cur_mean
    total_els += b
    return total_mean, total_els

def iterative_log_sum_exp(batch_tensor,total_sum,total_els, dim=0):
    b = batch_tensor.shape[dim]
    cur_sum = torch.logsumexp(batch_tensor, dim=0).sum()
    total_sum = torch.logsumexp(torch.stack( [total_sum,cur_sum] , dim=0), dim=0).sum()
    total_els += b  
    return total_sum,  total_els


def compute_nll(data_loader, model, device):
    model.eval()
    log_density = 0.
    M = 0
    for i, (data,target) in enumerate(data_loader): 
        with torch.no_grad():
            cur_log_density = - model.log_density(data.to(device)) 
            log_density, M = iterative_mean(cur_log_density, log_density,M)

    return log_density.mean()
