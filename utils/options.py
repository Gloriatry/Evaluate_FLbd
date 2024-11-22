#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
   parser = argparse.ArgumentParser()

   # basic arguments
   parser.add_argument('--init', type=str, default='None',
                     help="location of init model")
   parser.add_argument('--model_bank', action='store_true',
                     help="is it the process of training the pre-trained model")
   parser.add_argument('--gpu', type=int, default=0,
                     help="GPU ID, -1 for CPU")
   parser.add_argument('--seed', type=int, default=1,
                     help='random seed (default: 1)')
   parser.add_argument('--dataset', type=str,
                     default='mnist', help="name of dataset")
   parser.add_argument('--model', type=str,
                     default='Mnist_CNN', help='model name')

   # federated setting
   parser.add_argument('--epochs', type=int, default=500,
                     help="rounds of training")
   parser.add_argument('--num_users', type=int,
                     default=100, help="number of users: K")
   parser.add_argument('--frac', type=float, default=0.1,
                     help="the fraction of clients selected in each round")
   
   # learning setting
   parser.add_argument('--lr_m', type=float, default=0.05,
                     help="learning rate of malicious clients")
   parser.add_argument('--lr_b', type=float, default=0.01,
                     help="learning rate of benign clients")
   parser.add_argument('--momentum', type=float, default=0.9,
                     help="SGD momentum (default: 0.5)")
   parser.add_argument('--local_ep_m', type=int, default=6,
                     help="the number of local epochs for malicious clients")
   parser.add_argument('--local_ep_b', type=int, default=2,
                     help="the number of local epochs for benign clients")
   parser.add_argument('--local_bs', type=int, default=64,
                     help="local batch size when training")
   parser.add_argument('--bs', type=int, default=64, help="test batch size")

   # heterogenity setting
   parser.add_argument('--heter', type=str,
                     default='iid', help="heterogenity setting")
   parser.add_argument('--alpha', type=float, default=0.5,
                     help='degree of dirichlet distribution')
   parser.add_argument('--gau_noise', type=float, default=0, help='how much noise we add to some party')
   parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')

   # attack
   parser.add_argument('--malicious',type=float,default=0, help="proportion of mailicious clients")
   parser.add_argument('--attack', type=str,
                     default='badnet', help='attack method')
   parser.add_argument('--poison_frac', type=float, default=0.2, 
                     help="fraction of dataset to corrupt for backdoor attack, 1.0 for layer attack")
   
   parser.add_argument('--attack_label', type=int, default=5,
                        help="trigger for which label") # loan 7
   # attack_goal=-1 is all to one
   parser.add_argument('--attack_goal', type=int, default=-1,
                        help="trigger to which label")
   parser.add_argument('--trigger', type=str, default='square',
                     help="Kind of trigger")  
   # mnist 28*28  cifar10 32*32
   parser.add_argument('--triggerX', type=int, default='0',
                     help="position of trigger x-aix") 
   parser.add_argument('--triggerY', type=int, default='0',
                     help="position of trigger y-aix")
   
   parser.add_argument('--start_attack', type=int, default=200,
                        help="which epoch to start attack")
   
   
   # defense
   parser.add_argument('--defence', type=str,
                        default='avg', help="strategy of defence")
   parser.add_argument('--start_defence', type=int, default=200,
                        help="which epoch to start defence")
   # flshield
   parser.add_argument('--bijective', action='store_true')
   parser.add_argument('--clustering_methods', type=str, default=None)




   # save file 
   parser.add_argument('--save', type=str, default='save',
                     help="dic to save results (ending without /)")
   # parser.add_argument('--init', type=str, default='None',
   #                   help="location of init model")
   # federated arguments
   
   #***** badnet labelflip layerattack updateflip get_weight layerattack_rev layerattack_ER****
   

   # *****local_ep = 3, local_bs=50, lr=0.1*******

   # model arguments
   #*************************model******************************#
   # resnet cnn VGG mlp Mnist_2NN Mnist_CNN resnet20 rlr_mnist

   # other arguments
   #*************************dataset*******************************#
   # fashion_mnist mnist cifar
   
   
   

   # parser.add_argument('--iid', action='store_true',
   #                     help='whether i.i.d or not')

#************************atttack_label********************************#
   # --attack_begin 70 means accuracy is up to 70 then attack
   parser.add_argument('--attack_begin', type=int, default=0,
                     help="the accuracy begin to attack")
   
   parser.add_argument('--robustLR_threshold', type=int, default=4, 
                     help="break ties when votes sum to 0")
   
   parser.add_argument('--server_dataset', type=int,default=200,help="number of dataset in server")
   
   parser.add_argument('--server_lr', type=float,default=1,help="number of dataset in server using in fltrust")
   

   
   
   parser.add_argument('--split', type=str, default='user',
                     help="train-test split type, user or sample")   
   
   parser.add_argument('--verbose', action='store_true', help='verbose print')
   parser.add_argument('--wrong_mal', type=int, default=0)
   parser.add_argument('--right_ben', type=int, default=0)
   
   parser.add_argument('--mal_score', type=float, default=0)
   parser.add_argument('--ben_score', type=float, default=0)
   
   parser.add_argument('--turn', type=int, default=0)
   parser.add_argument('--noise', type=float, default=0.001)
   parser.add_argument('--all_clients', action='store_true',
                     help='aggregation over all clients') 


   args = parser.parse_args()
   return args
