### MAIN.py
### this file is modified from https://github.com/MichaelArbel/GeneralizedEBM/blob/master/main.py
### all credit goes to the author

from __future__ import print_function

#import torch
import argparse
import yaml
import torch
from trainer import Trainer

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def make_flags(args, config_file):
    if config_file:
        config = yaml.safe_load(open(config_file))
        dic = vars(args)
        all(map(dic.pop, config))
        dic.update(config)
    return args

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')


# log parameters
parser.add_argument('--log_dir', default='logs', type= str, help='log name ['']')
parser.add_argument('--log_name', type=str,   help='log directory for summaries and checkpoints ['']')

## change this for each run
parser.add_argument('--id', default="Gaus_crnn_gan_1", help='')
parser.add_argument('--d_path', default="/content/logs/train/cifar10/checkpoints_Gaus_crnn_gan_1_/d_best.pth", help='path to the trained energy network')
parser.add_argument('--g_path', default="/content/logs/train/cifar10/checkpoints_Gaus_crnn_gan_1_/g_best.pth", help='path to the trained base network')
parser.add_argument('--mode', type=str, default='train',   help='either "train" or "sample" ')
parser.add_argument('--dataset_type', default ='Stock' ,type= str , help='image dataset or others ["images","uci"]')
parser.add_argument('--features', default = 5, type= int,   help='features of time series')
#Sine, Stock 5 Energy 28 Chickpox 20

parser.add_argument('--seq_length', default = 32 ,type= int,   help='length of time series')
parser.add_argument('--generator', default = 'tts' ,type= str,  help='network type of the base : ["convolutional"]')
parser.add_argument('--discriminator', default = 'tts' ,type= str,  help='network type of the energy : ["convolutional"]')

### change this for crnn vs tts
parser.add_argument('--Z_dim', default = 100, type= int,  help='dimension of latent')
#parser.add_argument('--Z_dim', default = [32,20], type= int,  help='dimension of latent')

parser.add_argument('--checkpoint_freq', default = 250, type= int , help='frequency for saving checkpoints [1000]')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for the energy[.00001]')
parser.add_argument('--lr_generator', default=0.0001, type=float, help='learning rate for the base [.0002]')
parser.add_argument('--total_gen_iter',  default = 50000, type= float ,  help='total number of iterations for the base [150000]')

parser.add_argument('--gaus_phi', default=0.8, type=float, help='')
parser.add_argument('--gaus_sigma',  default=0.2, type= float ,  help='')


parser.add_argument('--data_path', default ='./data' ,type= str , help='directory to the dataset ["data"]')
parser.add_argument('--imagenet_train_path',  default ='' ,type= str ,  help='path to imagenet train set')
parser.add_argument('--imagenet_test_path',  default ='' ,type= str ,  help='path to imagenet test set')

parser.add_argument('--log_to_file', action='store_true',  help='log output in a file [False]')
parser.add_argument('--save_nothing', action='store_true', help="Disable saving of the model [False]")
parser.add_argument('--disp_freq', default = 50, type= int , help='frequency for displaying the loss [100]')


# control parameters

parser.add_argument('--train_mode', default='both', type=str,  help='either train both energy and base or only one of them ["both","energy","base"]')
parser.add_argument('--dataset', default='cifar10',type= str,  help='name of the dataset to use  ["cifar10","CelebA","Imagenet32","lsun"]')

# operational parameters
parser.add_argument('--device', default = 0 ,type= int,   help='gpu device [0]')
parser.add_argument('--seed', default = 0 ,type= int ,    help='seed for randomness [0]')
parser.add_argument('--dtype', default='32', type= str,   help='32 for float32 and 64 for float64 ["32"]')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers of the dataloader [4]')
#parser.add_argument('--dataparallel', action='store_true', help='parallelize over multiple gpus [False]')
parser.add_argument('--dataparallel', action='store_true', help='parallelize over multiple gpus [False]')
parser.add_argument('--slurm_id', default = '',type= str,  help='job id when using slurm, modified internally ['']')

parser.add_argument('--trainer_type', default ='default' ,type= str , help='the class for training / sampling ["default"]')

# model parameters
parser.add_argument('--g_model', default = 'resnet-sn' ,type= str,  help='architecture of the base network: ["dcgan","sngan"]')
parser.add_argument('--d_model', default = 'resnet-sn' ,type= str,  help='architecture of the energy network: ["dcgan","sngan"]')

parser.add_argument('--latent_noise', default ='gaussian' ,type= str , help='the distribution of latent noise ["gaussian"]')
parser.add_argument('--bn', action='store_true', help='batch-normalization [False]')
parser.add_argument('--num_blocks', default = 3, type= int, help='number of blocks for the NVP [3]')


# sampler parameters
parser.add_argument('--latent_sampler',  default ='langevin' ,type= str ,  help='which sampler to use ["langevin","mala","lmc","hmc"]')
parser.add_argument('--lmc_gamma', default=1e-06, type=float, help='step-size for the lmc sampler: [1e-2]')
parser.add_argument('--lmc_kappa', default=1e-2, type=float, help='friction parameter of the lmc sampler: ["100."]')
parser.add_argument('--num_sampler_steps', default = 1000, type= int,  help='number of sampler steps [100]')
parser.add_argument('--temperature', default = 100. ,type= float , help='temperature parameter [100]')

# batch sizes:
parser.add_argument('--fid_b_size', default=128, type= int,  help='batch-size for computing FID [128]')
parser.add_argument('--sample_b_size', default=128, type= int,  help='batch-size for sampling [1000]')
parser.add_argument('--b_size', default=128, type= int,  help='batch_size for training [128]')



# criterion
parser.add_argument('--criterion', default='lsgan',type= str, help='top level loss ["kale","donsker","cd","ml"]')
# regularization
parser.add_argument('--penalty_type', default = 'gradient',type= str,  help='the penalty for training the energy ["gradient_l2","gradient","l2","none"]')
parser.add_argument('--penalty_lambda', default=0.15, type=float, help='strenght of the penalty [.1]')
parser.add_argument('--initialize_log_partition', action='store_true', help='initialize log-partition using Monte-Carlo estimator [False]')
# training parameters

parser.add_argument('--total_epochs', default=100, type=int, help='total number of epochs [100]')
parser.add_argument('--n_iter_d_init', default = 1, type= int,  help='number of iterations on the energy at the begining of training and every 500 iterations on the base [100]')
parser.add_argument('--n_iter_d', default = 1, type= int,  help='number of iterations on the energy for every training iteration on the base [5]')



parser.add_argument('--noise_factor', default = 1, type= int,  help='factor multiplying the data batch size and giving the latent samples batch size [1]')


# optimizer parameters
parser.add_argument('--optimizer', default='Adam', type= str, help='Inner optimizer to compute the euclidean gradient["Adam"]')
parser.add_argument('--sgd_momentum', default=0., type=float, help='momentum parameter for SGD [0.]')
parser.add_argument('--beta_1', default=0.5, type=float, help='first parameter of Adam optimizer [.5]')
parser.add_argument('--beta_2', default=0.999, type=float, help='second parameter of Adam optimizer [.9]')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay [0.]')


# Scheduler parameters 
parser.add_argument('--use_scheduler', action='store_true', help='schedule the lr ["store_true"]')
parser.add_argument('--scheduler',  default ='ExponentialLR' ,type= str ,  help='scheduler ["MultiStepLR"]')
parser.add_argument('--milestone',  default = '10,50,70' ,type= str ,  help='decrease schedule for lr at epochs  ["10,50,70"]')
parser.add_argument('--scheduler_gamma',  default=0.8, type= float, help='decay of the learning rate [".8"]')
parser.add_argument('--lr_decay',  default = 0.8 ,type= float ,  help='decay of the learning rate [".8"]')

# metrics

parser.add_argument('--eval_kale', action='store_true', help='evaluate KALE on both training and test sets ["False"]')
parser.add_argument('--freq_kale', default = 2000, type= int , help='frequency for evaluating kale per iteratations [2000]')
parser.add_argument('--eval_fid', action='store_true', help='evaluate the FID scores [False]')
parser.add_argument('--fid_samples', default = 50000, type= int,  help='number of generated samples to evaluate the score [50000]')
parser.add_argument('--freq_fid', default = 2000, type= int, help='frequency for evaluating FID per iteratations [2000]')
parser.add_argument('--oldest_fid_iter', default = 20000, type= int, help='frequency for evaluating FID per iteratations [2000]')
parser.add_argument('--grad_clip', default = 1, type= int, help='frequency for evaluating FID per iteratations [2000]')
parser.add_argument('--no_trunc', action='store_true', help='evaluate the FID scores [False]')
parser.add_argument('--skipinit', action='store_true', help='calculate FID, but takes time')
parser.add_argument('--configs',  default ='' ,type= str ,  help='config file for the run ['']')

args = parser.parse_args("")

#args = parser.parse_args()
args = make_flags(args, args.configs)
if args.trainer_type=='default':
	trainer = Trainer(args)
trainer.main()
print('Finished!')







