### TRAINER.py

import math

import torch
import torch.nn as nn

import numpy as np

import csv
import sys
import os
import time
from datetime import datetime
import pprint
import socket
import json
import pickle as pkl
from torch.autograd import Variable
import pdb

import timeit

import helpers as hp
import compute as cp
import samplers
from utils import timer

import models

class Trainer(object):
    def __init__(self, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.args = args
        self.device = hp.assign_device(args.device)
        #self.run_id = str(round(time.time() % 1e7))
        self.run_id = args.id
        print(f"Run id: {self.run_id}")
        self.log_dir, self.checkpoint_dir, self.samples_dir = hp.init_logs(args, self.run_id, self.log_dir_formatter(args) )
        
        print(f"Process id: {str(os.getpid())} | hostname: {socket.gethostname()}")
        print(f"Run id: {self.run_id}")
        print(f"Time: {datetime.now()}")
        self.pp = pprint.PrettyPrinter(indent=4)
        self.pp.pprint(vars(args))
        print('==> Building model..')
        self.timer = Timer()
        self.mode = args.mode
        self.build_model()

        

    # model building functions
    def log_dir_formatter(self,args):
            return os.path.join(args.log_dir, args.mode, args.dataset)


    def build_model(self):
        self.train_loader, self.test_loader, self.valid_loader,self.input_dims = hp.get_data_loader(self.args,self.args.b_size, self.args.num_workers)
        
        self.generator = hp.get_base(self.args, self.input_dims, self.device)
        self.discriminator = hp.get_energy(self.args,self.input_dims, self.device)
        self.noise_gen = hp.get_latent_noise(self.args,self.args.Z_dim, self.device)
        self.fixed_latents = self.noise_gen.sample([64])
        self.eval_latents =torch.cat([ self.noise_gen.sample([self.args.sample_b_size]).cpu() for b in range(int(self.args.fid_samples/self.args.sample_b_size)+1)], dim=0)
        self.eval_latents = self.eval_latents[:self.args.fid_samples]
        self.assess_latents =torch.cat([ self.noise_gen.sample([self.args.sample_b_size]).cpu() for b in range(int(self.args.fid_samples/self.args.sample_b_size)+1)], dim=0)
        self.assess_latents = self.eval_latents[:self.args.fid_samples]
        self.eval_velocity =  torch.cat([ torch.zeros([self.args.sample_b_size, self.eval_latents.shape[1]]).cpu() for b in range(int(self.args.fid_samples/self.args.sample_b_size)+1)], dim=0)
        self.eval_velocity = self.eval_velocity[:self.args.fid_samples]
        # load models if path exists, define log partition if using kale and add to discriminator
        self.d_params = list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
        self.args.gaus_sigma = 1- self.args.gaus_sigma
        self.args.gaus_phi = 1- self.args.gaus_phi

        self.pred = 1


        #else:
        if self.args.criterion == 'kale':
              self.log_partition = nn.Parameter(torch.zeros(1).to(self.device))
              self.d_params.append(self.log_partition)
        else:
              self.log_partition = Variable(torch.zeros(1, requires_grad=False)).to(self.device)

        if self.mode == 'train':
            # optimizers
            self.optim_d = hp.get_optimizer(self.args, 'discriminator', self.d_params)
            self.optim_g = hp.get_optimizer(self.args, 'generator', self.generator.parameters())
            self.optim_partition = hp.get_optimizer(self.args, 'discriminator', [self.log_partition])
            # schedulers
            self.scheduler_d = hp.get_scheduler(self.args, self.optim_d)
            self.scheduler_g = hp.get_scheduler(self.args, self.optim_g)
            self.scheduler_partition = hp.get_scheduler(self.args, self.optim_partition)
            self.loss = hp.get_loss(self.args)
            self.counter = 0
            self.g_counter = 0
            self.g_loss = torch.tensor(0.)
            self.d_loss = torch.tensor(0.)

        if self.args.latent_sampler in ['imh', 'dot','spherelangevin']:
            self.latent_potential = samplers.Independent_Latent_potential(self.generator,self.discriminator,self.noise_gen) 
        elif self.args.latent_sampler in ['zero_temperature_langevin']:
            self.latent_potential = samplers.Cold_Latent_potential(self.generator,self.discriminator) 
        else:
            self.latent_potential = samplers.Latent_potential(self.generator,self.discriminator,self.noise_gen, self.args.temperature) 
        
        self.latent_sampler = hp.get_latent_sampler(self.args, self.latent_potential,self.args.Z_dim, self.device)

        dev_count = torch.cuda.device_count()    
        if self.args.dataparallel and dev_count>1 :
            self.generator = torch.nn.DataParallel(self.generator,device_ids=list(range(dev_count)))
            self.discriminator = torch.nn.DataParallel(self.discriminator,device_ids=list(range(dev_count)))
        self.accum_loss_g = []
        self.accum_loss_d = []
        self.true_train_scores = None
        self.true_valid_scores = None
        self.true_train_mu = None
        self.true_train_sigma = None
        self.true_valid_mu = None
        self.true_valid_sigma = None
        self.kids = None

    def main(self):
        print(f'==> Mode: {self.mode}')
        if self.mode == 'train':
            self.train()
            self.load_generator()
            self.generator.eval()
            self.load_discriminator()
            self.discriminator.eval()
            images = self.sample_images(self.eval_latents, self.args.sample_b_size)
            do_PCA_Analysis(self.args,images,f'PCA_Best_GAN', self.samples_dir)
            do_tSNE_Analysis(self.args,images,f'tSNE_Best_GAN', self.samples_dir)
            pred = get_predictive_score(self.args,images)
            print("Predictive score: " + str(self.pred))
            disc = get_discriminative_score(self.args,images)
            print( "Discriminator score: " + str(disc))
            self.sample()


        elif self.mode == 'eval':
            self.eval()
        elif self.mode == 'sample':
            self.load_generator()
            self.generator.eval()
            self.load_discriminator()
            self.discriminator.eval()
            self.sample()

    def load_generator(self):
        #g_path = os.path.join(self.checkpoint_dir, f'g_best.pth')
        #g_path = os.path.join(f'/content', g_path)
        g_model = torch.load(self.args.g_path, map_location=self.device)
        self.noise_gen = hp.get_normal(self.args.Z_dim, self.device)
        self.generator.load_state_dict(g_model)
        self.generator = self.generator.to(self.device)

    def load_discriminator(self):
        #d_path = os.path.join(self.checkpoint_dir, f'd_best.pth')
        d_model = torch.load(self.args.d_path, map_location= self.device )
        

        if self.args.criterion == 'kale':
            try:
                self.log_partition = d_model['log_partition'].to(self.device)
            except:
                self.log_partition = nn.Parameter(torch.zeros(1).to(self.device))
        d_model.pop('log_partition', None)
        self.discriminator.load_state_dict(d_model)
        self.discriminator = self.discriminator.to(self.device)

    #### FOR TRAINING THE NETWORK
    def train(self):
        done =False
        if self.args.initialize_log_partition:
            self.log_partition.data = self.init_log_partition()  
        while not done:
            self.train_epoch()
            if self.args.train_mode in ['both', 'base']:
                done =  self.g_counter >= self.args.total_gen_iter
            else:
                done =  self.counter >= self.args.total_gen_iter

    def train_epoch(self):

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device).clone().detach().to(torch.float32)
            self.counter += 1
            is_gstep, is_dstep = self.which_step()

            
            # discriminator takes n_iter_d steps of learning for each generator step
            if is_gstep:
                self.g_counter +=1
                self.g_loss = self.iteration(data, net_type='generator')
                self.accum_loss_g.append(self.g_loss.item())
            else:
                self.d_loss = self.iteration(data, net_type='discriminator')
                self.accum_loss_d.append(self.d_loss.item())
            
            if self.args.train_mode =='both':
                counter = self.g_counter
                is_valid_step = is_gstep
                if self.g_counter % self.args.disp_freq == 0 and is_gstep:
                    ag = np.asarray(self.accum_loss_g).mean()
                    ad = np.asarray(self.accum_loss_d).mean()
                    self.save_dictionary({'g_loss':ag, 'd_loss':ad, 'loss_iter': self.g_counter})
                    self.timer(self.g_counter, " base loss: %.8f, energy loss: %.8f" % ( ag, ad))
                    self.accum_loss_g = []
                    self.accum_loss_d = []

            elif self.args.train_mode =='base':
                counter = self.g_counter
                is_valid_step = is_gstep
                if self.g_counter % self.args.disp_freq == 0 and is_gstep:
                    ag = np.asarray(self.accum_loss_g).mean()
                    self.save_dictionary({'g_loss':ag, 'loss_iter': self.g_counter})
                    self.timer(self.g_counter, " base loss: %.8f" % ag)
                    self.accum_loss_g = []
            
            elif self.args.train_mode =='energy':
                counter = self.counter
                is_valid_step = is_dstep
                if self.counter % self.args.disp_freq == 0 and is_dstep:
                    ad = np.asarray(self.accum_loss_d).mean()
                    self.save_dictionary({'d_loss':ad, 'loss_iter': self.counter})
                    self.timer(self.counter, " energy loss: %.8f" % ad)
                    self.accum_loss_d = []

            if counter % self.args.checkpoint_freq == 0 and is_valid_step:
                if self.args.train_mode in ['both', 'base'] and self.args.dataset_type in ['Sine','Stock','Energy','Gaus','Chickenpox']:
                    images = self.sample_images(self.eval_latents, self.args.sample_b_size)
                    if self.args.dataset_type != "Gaus":
                      make_and_save_ts_images(images, f'Iter_{str(self.g_counter).zfill(3)}', self.samples_dir)
                      do_PCA_Analysis(self.args,images,f'PCA_Iter_{str(self.g_counter).zfill(3)}', self.samples_dir)
                      do_tSNE_Analysis(self.args,images,f'tSNE_Iter_{str(self.g_counter).zfill(3)}', self.samples_dir)
                    pred = get_predictive_score(self.args,images)
                    print(pred)
                    disc = get_discriminative_score(self.args,images)
                    print( "Discriminator score: " + str(disc))
                    if pred['MAE'][1] < self.pred:
                    #if pred < self.pred:
                      self.save_checkpoint(self.g_counter,best = True)
                      self.pred = pred['MAE'][1]

                      


            if (counter % 20000 == 0 and is_valid_step):
                print('decreasing lr')
                self.scheduler_d.step()
                self.scheduler_g.step()   


            self.eval_fid = is_gstep and np.mod(self.g_counter, self.args.freq_fid)==0  and self.args.eval_fid
            self.eval_kale = is_valid_step and np.mod(counter,self.args.freq_kale)==0  and self.args.eval_kale
            self.eval()

    def eval(self):
        if self.eval_fid or self.eval_kale:
            images = self.sample_images(self.eval_latents, self.args.sample_b_size, as_list=True)
            if self.eval_kale:
                KALE_train, base_mean, log_partition = self.compute_kale(self.train_loader, images)
                KALE_test, _ , _ = self.compute_kale(self.test_loader,  images, precomputed_stats = (base_mean,log_partition) )
                self.save_dictionary({'kale_train':KALE_train.item(), 'kale_test':KALE_test.item(), 'base_mean':base_mean.item(), 'log_partition':log_partition.item(), 'kale_iter':self.counter})
            if self.eval_fid:
            
                images = torch.split( torch.cat(images, dim=0), self.args.fid_b_size, dim=0)
                fid_train, fid_test = self.compute_fid( images, loader_types = ['train','valid'])
                self.fid_train = fid_train
                self.save_dictionary({'fid_train':fid_train, 'fid_test':fid_test, 'kid_train':self.kids['kid_train'],'kid_valid':self.kids['kid_valid'], 'fid_iter':self.g_counter})


    def which_step(self):
        if self.args.train_mode =='both':
            if self.g_counter < 2 or  (self.g_counter%500==0) :
                n_iter_d = self.args.n_iter_d_init
            else:
                n_iter_d = self.args.n_iter_d
            is_gstep = (np.mod(self.counter, n_iter_d+1) == 0) and (self.counter > self.args.n_iter_d_init)

            return is_gstep, ~is_gstep
        elif self.args.train_mode =='base':
            return True, False
        elif self.args.train_mode =='energy':
            return False, True

    # take a step, and maybe train either the discriminator or generator. also used in eval
    def iteration(self, data, net_type, train_mode=True):
        optimizer = self.prepare_optimizer(net_type)
        # get data and run through discriminator
        if not self.args.generator == "crnn":
            Z = self.noise_gen.sample([self.args.noise_factor*data.shape[0]])
        else:
            Z = torch.empty([data.shape[0], self.args.seq_length, self.args.features]).uniform_()
            g_states = self.generator.init_hidden(data.shape[0])
            d_state = self.discriminator.init_hidden(data.shape[0])

        with_gen_grad = train_mode and (net_type=='generator')

        with torch.set_grad_enabled(with_gen_grad) and torch.backends.cudnn.flags(enabled=False):
            if not self.args.generator == "crnn":
                fake_data = self.generator(Z)
            else:
                fake_data,_ = self.generator(Z,g_states)

        with torch.set_grad_enabled(train_mode) and torch.backends.cudnn.flags(enabled=False):
            if not self.args.generator == "crnn":
                true_results = self.discriminator(data)
                fake_results = self.discriminator(fake_data)
            else:
                true_results,_,_ = self.discriminator(data,d_state)
                fake_results,_,_ = self.discriminator(fake_data,d_state)

            log_partition, batch_log_partition = self.compute_log_partition(fake_results, net_type ,with_batch_est=True)

        if self.args.criterion in ['kale','donsker']:
            true_results = true_results + log_partition
            fake_results = fake_results + log_partition
        # calculate loss and propagate
        loss = self.loss(true_results, fake_results, net_type)

        if train_mode:
            total_loss = self.add_penalty(loss, net_type, data, fake_data)
            #if not self.args.generator == "crnn":
            total_loss.backward()
            if self.args.grad_clip>0:
                self.grad_clip(optimizer, net_type=net_type)
            optimizer.step()
         return loss

    def prepare_optimizer(self,net_type):
        if net_type=='discriminator':           
            optimizer = self.optim_d
            self.discriminator.train()
            self.generator.eval()
        elif net_type=='generator':
            optimizer = self.optim_g
            self.generator.train()
            self.discriminator.eval()  
        optimizer.zero_grad()
        return optimizer

    def add_penalty(self,loss, net_type, data, fake_data):
        if net_type=='discriminator':
            penalty = self.args.penalty_lambda * cp.penalty_d(self.args, self.discriminator, data, fake_data, self.device)
            total_loss = loss + penalty
        else:
            total_loss = loss
        return total_loss

    def init_log_partition(self):
        log_partition = torch.tensor(0.).to(self.device)
        M = 0
        num_batches = 100
        self.generator.eval()
        self.discriminator.eval()
        for batch_idx in range(num_batches):
            with torch.no_grad():
                Z = self.noise_gen.sample([self.args.sample_b_size])
                fake_data = self.generator(Z)
                fake_data = -self.discriminator(fake_data)
                log_partition,M = iterative_log_sum_exp(fake_data,log_partition,M)
        log_partition = log_partition - np.log(M)
        return torch.tensor(log_partition.item()).to(self.device)

    def compute_log_partition(self,fake_results, net_type, with_batch_est = False):
        batch_log_partition = torch.logsumexp(-fake_results, dim=0)- np.log(fake_results.shape[0])
        batch_log_partition = cp.batch_log_partition.squeeze()
        val_log_partition = self.log_partition
        tmp = fake_results + val_log_partition

        if net_type=='discriminator':
            if self.args.criterion=='donsker':
                log_partition = batch_log_partition.detach()
            else:
                log_partition = val_log_partition
        else:
            log_partition = batch_log_partition
        if with_batch_est:
            return log_partition, batch_log_partition
        else:
            return log_partition

    def grad_clip(self,optimizer, net_type='discriminator'):
        if net_type=='discriminator':
            params = self.d_params[:-1]
            for i, param in enumerate(params):
                new_grad = 2.*(param.grad.data)/(1+ (param.grad.data)**2)
                if math.isfinite(torch.norm(new_grad).item()):
                    param.grad.data = 1.*new_grad
                else:
                    print('nan grad')
                    param.grad.data = torch.zeros_like(new_grad)

            param = self.d_params[-1]
            new_grad = param.grad.data/(1-param.grad.data)
            if math.isfinite(torch.norm(new_grad).item()):
                param.grad.data = new_grad
            else:
                param.grad.data = torch.zeros_like(new_grad)

    #### FOR EVALUATING PERFORMANCEs

    # evaluate a pretrained model thoroughly via FID

    def compute_kale(self,data_loader,base_loader, precomputed_stats=None):
        self.discriminator.eval()
        base_mean = torch.tensor(0.).to(self.device)
        data_mean = 0
        if precomputed_stats is None:
            M = 0
            with torch.no_grad():
                for img in base_loader:
                    energy  = -self.discriminator(img.to(self.device))
                    if self.args.criterion == 'donsker':
                        base_mean,M = cp.iterative_log_sum_exp(torch.exp(energy),base_mean,M)
                    else:
                        energy = -torch.exp(energy - self.log_partition ) 
                        base_mean, M = cp.iterative_mean(energy, base_mean,M)
            if self.args.criterion=='donsker':
                log_partition = 1.*base_mean -np.log(M)
                base_mean = torch.tensor(-1.).to(self.device)
            else:
                log_partition = self.log_partition

        else:
            base_mean, log_partition= precomputed_stats 

        M = 0
        for data, target in data_loader: 
            with torch.no_grad():
                data_energy = -(self.discriminator(data.to(self.device)) + log_partition)
            data_mean, M = cp.iterative_mean(data_energy, data_mean,M)

        KALE = data_mean + base_mean + 1
        return KALE, base_mean, log_partition

    #### FOR Sampling
    #### FOR Sampling
    def init_latents(self):
        if self.args.latent_sampler == 'dot':
            priors = 1.*self.eval_latents.unsqueeze(-1).clone()
            max_samples = np.minimum(500, self.eval_latents.shape[0])
            self.latent_sampler.estimate_lip(self.eval_latents[:max_samples].to(self.device))
            out = torch.cat([self.eval_latents.unsqueeze(-1), priors ], dim=-1)
            return out
        elif self.args.latent_sampler == 'lmc':
            out = torch.cat([self.eval_latents.unsqueeze(-1), self.eval_velocity.unsqueeze(-1) ], dim=-1)
            return out
        else:
            return self.eval_latents
    def get_posterior(self,posteriors):
        if self.args.latent_sampler=='dot':
            return posteriors[:,:,0]
        elif self.args.latent_sampler=='lmc':
            return posteriors[:,:,0]
        else:
            return posteriors

    def sample(self):
        T = 10
        max_saved = 50
        num_dps = int(self.args.num_sampler_steps/T)+1
        start = time.time()
        predd = 1
        discc = 1
        for i in range(num_dps):
            iter_num = i*T
            if i==0: 
                posteriors = self.init_latents()
            else:
                posteriors = self.sample_latents(posteriors, self.args.sample_b_size , T)

            images = self.sample_images(self.get_posterior(posteriors),self.args.fid_b_size, as_list=True)
            images = torch.cat(images, dim=0)
            saved_images = images[:64]
            saved_posteriors = posteriors[:64]
            end = time.time()
            start = end
            if  i%1==0:
                make_and_save_ts_images(images, f'Sample_{str(i).zfill(3)}', self.samples_dir)
                do_PCA_Analysis(self.args,images,f'PCA_Iter_{str(i).zfill(3)}', self.samples_dir)
                do_tSNE_Analysis(self.args,images,f'tSNE_Iter_{str(i).zfill(3)}', self.samples_dir)
                pred = get_predictive_score(self.args,images)
                disc = get_discriminative_score(self.args,images)
                print( "Discriminator sample score: " + str(i) + " "+ str(disc))
                print( "Predictive sample score: " + str(i) + " "+ str(pred))
                if pred["MAE"][1] < predd:
                  do_PCA_Analysis(self.args,images,f'PCA_Best', self.samples_dir)
                  do_tSNE_Analysis(self.args,images,f'tSNE_Best', self.samples_dir)
                  predd = pred["MAE"][1]
                  if disc < discc:
                    discc = disc
            if i%20==0 and i>0:
                self.latent_sampler.gamma *= 0.5
                print(f'decreasing lr for sampling: {self.latent_sampler.gamma}')
        print("Prediction " + str(predd) + " Discriminative " + str(discc))


        

    def sample_latents(self,priors,b_size, T, with_acceptance = False):
        avg_time = 0
        posteriors = []
        avg_acceptences = []
        for b, prior in enumerate(priors.split(b_size, dim=0)):
            st = time.time()
            prior = prior.clone().to(self.device)
            posterior,avg_acceptence = self.latent_sampler.sample(prior,sample_chain=False,T=T)            
            posteriors.append(posterior)
            avg_acceptences.append(avg_acceptence)

        posteriors = torch.cat(posteriors, axis=0)
        avg_acceptences = np.mean(np.array(avg_acceptences), axis=0)

        if with_acceptance:
            return posteriors, avg_acceptences
        else:
            return posteriors

    def sample_images(self, latents, b_size =128, to_cpu = True, as_list=False):
        self.discriminator.eval()
        self.generator.eval()
        images = []
        if self.args.generator != "crnn":
          for latent in latents.split(b_size, dim=0):
            with torch.no_grad():
                  img = self.generator(latent.to(self.device))
            if to_cpu:
                img = img.cpu()
            images.append(img)
        else:
          for latent in latents.split(b_size, dim=0):
            
            with torch.no_grad():
                if self.args.generator == "crnn":
                  g_states = self.generator.init_hidden(latent.shape[0])
                  img,_ = self.generator(latent.to(self.device),g_states)
                else:
                  img = self.generator(latent.to(self.device))
            if to_cpu:
                img = img.cpu()
            images.append(img)
        if as_list:
            return images
        else:
            return torch.cat(images, dim=0)

### Savers

    def save_checkpoint(self, epoch,best=False):
        if self.args.save_nothing:
            return
        if self.args.train_mode in ['both', 'energy']:
            d_dict = self.discriminator.state_dict()
            if self.args.criterion == 'kale':
                d_dict['log_partition'] = self.log_partition
            if best:
                d_path = os.path.join(self.checkpoint_dir, f'd_best.pth')
            else:
                d_path = os.path.join(self.checkpoint_dir, f'd_{epoch}.pth')
            torch.save(d_dict, d_path)
            print(f'Saved {d_path}')
        if self.args.train_mode in ['both', 'base']:
            if best:
                g_path = os.path.join(self.checkpoint_dir, f'g_best.pth')
            else:    
                g_path = os.path.join(self.checkpoint_dir, f'g_{epoch}.pth')
            torch.save(self.generator.state_dict(), g_path)
            print(f'Saved {g_path}')

    # just evaluate the performance (via KALE metric) during training
    
    def save_dictionary(self,new_dict, dic_arrays = None ,index=0 ):
        if dic_arrays is not None:
            fname = os.path.join(self.samples_dir, f'MCMC_samples_{str(index).zfill(3)}.pkl')
            np.savez(fname, **dic_arrays)
            new_dict['index'] = index
            new_dict['path_arrays'] = fname
        file_name = os.path.join(self.samples_dir, f'stats_seed_{self.args.seed}')
        with open(file_name+'.json','a') as f:
            json.dump(new_dict,f)
            f.write(os.linesep)


