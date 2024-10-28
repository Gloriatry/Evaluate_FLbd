#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset_label, num_clients, num_classes, q):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    dict_users = non_iid_distribution_client(proportion, num_clients, num_classes)
    #  output clients' labels information
    # check_data_each_client(dataset_label, dict_users, num_clients, num_classes)
    return dict_users

# def non_iid_distribution_group(dataset_label, num_clients, num_classes, q):
#     dict_users, all_idxs = {}, [i for i in range(len(dataset_label))]
#     for i in range(num_classes):
#         dict_users[i] = set([])
#     for k in range(num_classes):
#         idx_k = np.where(dataset_label == k)[0]
#         num_idx_k = len(idx_k)
        
#         selected_q_data = set(np.random.choice(idx_k, int(num_idx_k*q) , replace=False))
#         dict_users[k] = dict_users[k]|selected_q_data
#         idx_k = list(set(idx_k) - selected_q_data)
#         all_idxs = list(set(all_idxs) - selected_q_data)
#         for other_group in range(num_classes):
#             if other_group == k:
#                 continue
#             selected_not_q_data = set(np.random.choice(idx_k, int(num_idx_k*(1-q)/(num_classes-1)) , replace=False))
#             dict_users[other_group] = dict_users[other_group]|selected_not_q_data
#             idx_k = list(set(idx_k) - selected_not_q_data)
#             all_idxs = list(set(all_idxs) - selected_not_q_data)
#     print(len(all_idxs),' samples are remained')
#     print('random put those samples into groups')
#     num_rem_each_group = len(all_idxs) // num_classes
#     for i in range(num_classes):
#         selected_rem_data = set(np.random.choice(all_idxs, num_rem_each_group, replace=False))
#         dict_users[i] = dict_users[i]|selected_rem_data
#         all_idxs = list(set(all_idxs) - selected_rem_data)
#     print(len(all_idxs),' samples are remained after relocating')
#     return dict_users

# def non_iid_distribution_client(group_proportion, num_clients, num_classes):
#     num_each_group = num_clients // num_classes
#     num_data_each_client = len(group_proportion[0]) // num_each_group
#     dict_users, all_idxs = {}, [i for i in range(num_data_each_client*num_clients)]
#     for i in range(num_classes):
#         group_data = list(group_proportion[i])
#         for j in range(num_each_group):
#             selected_data = set(np.random.choice(group_data, num_data_each_client, replace=False))
#             dict_users[i*10+j] = selected_data
#             group_data = list(set(group_data) - selected_data)
#             all_idxs = list(set(all_idxs) - selected_data)
#     print(len(all_idxs),' samples are remained')
#     return dict_users
# def check_data_each_client(dataset_label, client_data_proportion, num_client, num_classes):
#     for client in client_data_proportion.keys():
#         client_data = dataset_label[list(client_data_proportion[client])]
#         print('client', client, 'distribution information:')
#         for i in range(num_classes):
#             print('class ', i, ':', len(client_data[client_data==i])/len(client_data))

# def iid_split(dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

def homo(dataset_train, n_parties):
    n_train = len(dataset_train)
    idxs = np.random.permutation(n_train)
    batch_idxs = np.array_split(idxs, n_parties)
    net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    return net_dataidx_map

def one_label_expert(dataset_label, num_clients, num_classes, q):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    dict_users = non_iid_distribution_client(proportion, num_clients, num_classes)
    #  output clients' labels information
    # check_data_each_client(dataset_label, dict_users, num_clients, num_classes)
    return dict_users

def non_iid_distribution_group(dataset_label, num_clients, num_classes, q):
    dict_users, all_idxs = {}, [i for i in range(len(dataset_label))]
    for i in range(num_classes):
        dict_users[i] = set([])
    for k in range(num_classes):
        idx_k = np.where(dataset_label == k)[0]
        num_idx_k = len(idx_k)
        
        selected_q_data = set(np.random.choice(idx_k, int(num_idx_k*q) , replace=False))
        dict_users[k] = dict_users[k]|selected_q_data
        idx_k = list(set(idx_k) - selected_q_data)
        all_idxs = list(set(all_idxs) - selected_q_data)
        for other_group in range(num_classes):
            if other_group == k:
                continue
            selected_not_q_data = set(np.random.choice(idx_k, int(num_idx_k*(1-q)/(num_classes-1)) , replace=False))
            dict_users[other_group] = dict_users[other_group]|selected_not_q_data
            idx_k = list(set(idx_k) - selected_not_q_data)
            all_idxs = list(set(all_idxs) - selected_not_q_data)
    print(len(all_idxs),' samples are remained')
    print('random put those samples into groups')
    num_rem_each_group = len(all_idxs) // num_classes
    for i in range(num_classes):
        selected_rem_data = set(np.random.choice(all_idxs, num_rem_each_group, replace=False))
        dict_users[i] = dict_users[i]|selected_rem_data
        all_idxs = list(set(all_idxs) - selected_rem_data)
    print(len(all_idxs),' samples are remained after relocating')
    return dict_users

def non_iid_distribution_client(group_proportion, num_clients, num_classes):
    num_each_group = num_clients // num_classes
    # num_data_each_client = len(group_proportion[0]) // num_each_group
    # dict_users, all_idxs = {}, [i for i in range(num_data_each_client*num_clients)]
    dict_users = {}
    for i in range(num_classes):
        group_data = list(group_proportion[i])
        num_data_each_client = len(group_proportion[i]) // num_each_group
        for j in range(num_each_group):
            selected_data = set(np.random.choice(group_data, num_data_each_client, replace=False))
            dict_users[i*num_each_group+j] = selected_data
            group_data = list(set(group_data) - selected_data)
            # all_idxs = list(set(all_idxs) - selected_data)
    # print(len(all_idxs),' samples are remained')
    return dict_users

# def dirichlet(dataset_train, no_participants, alpha):
#     cifar_classes = {}
#     for ind, x in enumerate(dataset_train):  # for cifar: 50000; for tinyimagenet: 100000
#         _, label = x
#         if label in cifar_classes:
#             cifar_classes[label].append(ind)
#         else:
#             cifar_classes[label] = [ind]
#     class_size = len(cifar_classes[0])  # for cifar: 5000
#     per_participant_list = {i: [] for i in range(no_participants)}
#     no_classes = len(cifar_classes.keys())  # for cifar: 10

#     image_nums = []
#     for n in range(no_classes):
#         image_num = []
#         random.shuffle(cifar_classes[n])
#         sampled_probabilities = class_size * np.random.dirichlet(
#             np.array(no_participants * [alpha]))
#         for user in range(no_participants):
#             no_imgs = int(round(sampled_probabilities[user]))
#             sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
#             image_num.append(len(sampled_list))
#             per_participant_list[user].extend(sampled_list)
#             cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
#         image_nums.append(image_num)
#     # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
#     return per_participant_list

def dirichlet(dataset_train, no_participants, alpha):
    min_size = 0
    min_require_size = 10
    # K = 10
    # if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
    #     K = 2
    #     # min_require_size = 100
    # if dataset == 'cifar100':
    #     K = 100
    # elif dataset == 'tinyimagenet':
    #     K = 200
    data_classes = {}
    for ind, x in enumerate(dataset_train):
        _, label = x
        if label in data_classes:
            data_classes[label].append(ind)
        else:
            data_classes[label] = [ind]
    class_size = len(data_classes[0])
    K = len(data_classes.keys())  # for cifar: 10

    N = len(dataset_train)
    #np.random.seed(2020)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(no_participants)]
        for k in range(K):
            # idx_k = np.where(y_train == k)[0]
            idx_k = data_classes[k]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, no_participants))
            # logger.info("proportions1: ", proportions)
            # logger.info("sum pro1:", np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / no_participants) for p, idx_j in zip(proportions, idx_batch)])
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            # if K == 2 and n_parties <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break


    for j in range(no_participants):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    
    return net_dataidx_map

def label_num_noniid(heter, dataset_train, n_parties):
    num = eval(heter[13:])
    print(f"now split in noniid-#label{num} mode")
    data_classes = {}
    for ind, x in enumerate(dataset_train):
        _, label = x
        if label in data_classes:
            data_classes[label].append(ind)
        else:
            data_classes[label] = [ind]
    K = len(data_classes.keys())
    if num == 10:
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
        for i in range(10):
            idx_k = data_classes[i]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k,n_parties)
            for j in range(n_parties):
                net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
    else:
        times=[0 for i in range(K)]
        contain=[]
        for i in range(n_parties):
            current=[i%K]
            times[i%K]+=1
            j=1
            while (j<num):
                ind=random.randint(0,K-1)
                if (ind not in current):
                    j=j+1
                    current.append(ind)
                    times[ind]+=1
            contain.append(current)
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
        for i in range(K):
            idx_k = data_classes[i]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k,times[i])
            ids=0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                    ids+=1
        
    return net_dataidx_map

def quantity_noniid(dataset_train, n_parties, alpha):
    n_train = len(dataset_train)
    idxs = np.random.permutation(n_train)
    min_size = 0
    while min_size < 10:
        proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
        proportions = proportions/proportions.sum()
        min_size = np.min(proportions*len(idxs))
    proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
    batch_idxs = np.split(idxs,proportions)
    net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    return net_dataidx_map

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
