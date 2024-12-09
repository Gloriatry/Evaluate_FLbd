# -*- coding = utf-8 -*-
import numpy as np
import torch
import copy
import time
import hdbscan
import os
import matplotlib.pyplot as plt  
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA
import sklearn.metrics.pairwise as smp
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset
from utils.CrowdGuard import CrowdGuardClientValidation
from utils.load_data import load_data
from flshield_utils.cluster_grads import cluster_grads as cluster_function
from flshield_utils.validation_test import validation_test
from flshield_utils.impute_validation import impute_validation
from flshield_utils.validation_processing import ValidationProcessor
from tqdm import tqdm

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # print(item)
        image, label = self.dataset[self.idxs[item]]
        return image, label


def cos(a, b):
    # res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-
    res = (np.dot(a, b) + 1e-9) / (np.linalg.norm(a) + 1e-9) / \
        (np.linalg.norm(b) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res


def fltrust(update_params, central_param, global_model, args, writer, iter):
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients

    FLTrustTotalScore = 0
    score_list = []
    central_param_v = parameters_dict_to_vector_flt(central_param)
    central_norm = torch.norm(central_param_v)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to(args.device)
    sum_parameters = None
    benign_client = []
    for i, local_parameters in enumerate(update_params):
        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)
        # 计算cos相似度得分和向量长度裁剪值
        client_cos = cos(central_param_v, local_parameters_v)
        # client_cos = max(client_cos.item(), 0)
        if client_cos >= 0:
            benign_client.append(i)
        # client_clipped_value = central_norm/torch.norm(local_parameters_v)
        # score_list.append(client_cos)
        # FLTrustTotalScore += client_cos
    #     if sum_parameters is None:
    #         sum_parameters = {}
    #         for key, var in local_parameters.items():
    #             # 乘得分 再乘裁剪值
    #             sum_parameters[key] = client_cos * \
    #                 client_clipped_value * var.clone()
    #     else:
    #         for var in sum_parameters:
    #             sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * local_parameters[
    #                 var]
    # print(score_list)

    for i in range(len(benign_client)):
        gama = central_norm/torch.norm(parameters_dict_to_vector_flt(update_params[benign_client[i]]))
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)

    record_TNR_TPR(args, benign_client, writer, iter)
    # args.psum += num_clients - len(benign_client)
    # args.nsum += len(benign_client)
    # fn = 0
    # for i in range(len(benign_client)):
    #     if benign_client[i] < num_malicious_clients:
    #         fn += 1
    #     else:
    #         args.tn += 1
    # args.tp += num_malicious_clients - fn
    # if args.psum == 0:
    #     args.psum += 1e-10
    # TPR = args.tp / args.psum
    # TNR = args.tn / args.nsum
    # writer.add_scalar("Metric/TPR", TPR, iter)
    # writer.add_scalar("Metric/TNR", TNR, iter)

    # if FLTrustTotalScore == 0:
    #     print(score_list)
    #     return global_parameters
    # for var in global_parameters:
    #     # 除以所以客户端的信任得分总和
    #     temp = (sum_parameters[var] / FLTrustTotalScore)
    #     if global_parameters[var].type() != temp.type():
    #         temp = temp.type(global_parameters[var].type())
    #     if var.split('.')[-1] == 'num_batches_tracked':
    #         global_parameters[var] = params[0][var]
    #     else:
    #         global_parameters[var] += temp * args.server_lr
    # print(score_list)
    return global_model


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector_flt_cpu(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.cpu().view(-1))
    return torch.cat(vec)


def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    if sum_parameters is None:
        return global_parameters
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters

def no_defence_weight(params, global_parameters, weights):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone() * weights[i]
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var] * weights[i]
    if sum_parameters is None:
        return global_parameters
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / sum(weights))

    return global_parameters

def multi_krum(gradients, n_attackers, args, multi_k=False):

    grads = flatten_grads(gradients)

    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        scores = None
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        print(scores)
        args.krum_distance.append(scores)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    # aggregate = torch.mean(candidates, dim=0)

    # return aggregate, np.array(candidate_indices)
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    args.turn+=1
    if multi_k == False:
        if candidate_indices[0] < num_malicious_clients:
            args.wrong_mal += 1
            
    print(candidate_indices)
    
    print('Proportion of malicious are selected:'+str(args.wrong_mal/args.turn))

    for i in range(len(scores)):
        if i < num_malicious_clients:
            args.mal_score += scores[i]
        else:
            args.ben_score += scores[i]
    
    return np.array(candidate_indices)



def flatten_grads(gradients):

    param_order = gradients[0].keys()

    flat_epochs = []

    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in param_order:
            try:
                user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
            except:
                user_arr.extend(
                    [grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs




def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        update2[key] = update[key] - model[key]
    return update2



def RLR(global_model, agent_updates_list, args):
    """
    agent_updates_dict: dict['key']=one_dimension_update
    agent_updates_list: list[0] = model.dict
    global_model: net
    """
    # args.robustLR_threshold = 6
    args.server_lr = 1

    grad_list = []
    for i in agent_updates_list:
        grad_list.append(parameters_dict_to_vector_rlr(i))
    agent_updates_list = grad_list
    

    aggregated_updates = 0
    for update in agent_updates_list:
        # print(update.shape)  # torch.Size([1199882])
        aggregated_updates += update
    aggregated_updates /= len(agent_updates_list)
    lr_vector = compute_robustLR(agent_updates_list, args)
    cur_global_params = parameters_dict_to_vector_rlr(global_model.state_dict())
    new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
    global_w = vector_to_parameters_dict(new_global_params, global_model.state_dict())
    # print(cur_global_params == vector_to_parameters_dict(new_global_params, global_model.state_dict()))
    return global_w

def parameters_dict_to_vector_rlr(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)



def vector_to_parameters_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict

def compute_robustLR(params, args):
    agent_updates_sign = [torch.sign(update) for update in params]  
    sm_of_signs = torch.abs(sum(agent_updates_sign))
    # print(len(agent_updates_sign)) #10
    # print(agent_updates_sign[0].shape) #torch.Size([1199882])
    sm_of_signs[sm_of_signs < args.robustLR_threshold] = -args.server_lr
    sm_of_signs[sm_of_signs >= args.robustLR_threshold] = args.server_lr 
    return sm_of_signs.to(args.gpu)
   
def record_TNR_TPR(args, benign_client:list, writer, iter):
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients

    args.psum += num_clients - len(benign_client)
    args.nsum += len(benign_client)
    fn = 0
    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            fn += 1
        else:
            args.tn += 1
    args.tp += num_malicious_clients - fn
    if args.psum == 0:
        args.psum += 1e-10
    TPR = args.tp / args.psum
    TNR = args.tn / args.nsum
    writer.add_scalar("Metric/TPR", TPR, iter)
    writer.add_scalar("Metric/TNR", TNR, iter)
    writer.add_scalar("Metric/TP", args.tp / (num_malicious_clients*(iter-args.start_defence)), iter)
    writer.add_scalar("Metric/TN", args.tn / (num_benign_clients*(iter-args.start_defence)), iter)


def flame(local_model, update_params, global_model, args, writer, file_name, iter):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to(args.device)
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
        for i in range(len(local_model_vector)):
            # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    # print(benign_client)
   
    # for i in range(len(benign_client)):
    #     if benign_client[i] < num_malicious_clients:
    #         args.wrong_mal+=1
    #     else:
    #         #  minus per benign in cluster
    #         args.right_ben += 1
    # args.turn+=1
    # print('proportion of malicious are selected:',args.wrong_mal/(num_malicious_clients*args.turn))
    # print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))

    # visualize the cos vector and results of hdbscan
    if iter > args.start_attack and iter % 100 == 0:
        file_path = "/root/project/epics/" + file_name
        if not os.path.exists(file_path):    
            os.makedirs(file_path)
        cos_proj = PCA(n_components=2).fit_transform(cos_list)
        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
        color_map = ['r'] * num_malicious_clients + ['b'] * num_benign_clients
        marker_map = ['o' if x >= 0 else '^' for x in clusterer.labels_]
        for i in range(num_clients):
            ax.scatter(cos_proj[i, 0], cos_proj[i, 1], c=color_map[i], marker=marker_map[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('PCA Axis 1', fontsize=17)
        ax.set_ylabel('PCA Axis 2', fontsize=17)
        plt.savefig(os.path.join(file_path, str(iter)+'.pdf'))

    record_TNR_TPR(args, benign_client, writer, iter)
    # args.psum += num_clients - len(benign_client)
    # args.nsum += len(benign_client)
    # fn = 0
    # for i in range(len(benign_client)):
    #     if benign_client[i] < num_malicious_clients:
    #         fn += 1
    #     else:
    #         args.tn += 1
    # args.tp += num_malicious_clients - fn
    # TPR = args.tp / args.psum
    # TNR = args.tn / args.nsum
    # writer.add_scalar("Metric/TPR", TPR, iter)
    # writer.add_scalar("Metric/TNR", TNR, iter)
    
    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value/norm_list[benign_client[i]]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
    #add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
                    continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0,std=args.noise*clip_value)
        var += temp
    return global_model

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

def multi_metrics(net_list, update_params, global_model, args, writer, iter):
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients

    vectorize_nets = [vectorize_net(cm).detach() for cm in net_list]
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to(args.device)
    cos_dis = [0.0] * len(vectorize_nets)
    length_dis = [0.0] * len(vectorize_nets)
    manhattan_dis = [0.0] * len(vectorize_nets)
    for i, g_i in enumerate(vectorize_nets):
        for j in range(len(vectorize_nets)):
            if i != j:
                g_j = vectorize_nets[j]

                # cosine_distance = float(
                #     (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                # manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                # length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                cosine_distance = (1 - cos(g_i, g_j)) ** 2  
                manhattan_distance = torch.norm(g_i - g_j, p=1).item()
                length_distance = torch.abs(torch.norm(g_i) - torch.norm(g_j)).item()

                cos_dis[i] += cosine_distance
                length_dis[i] += length_distance
                manhattan_dis[i] += manhattan_distance
    
    tri_distance = torch.tensor([cos_dis, manhattan_dis, length_dis], dtype=torch.float32).T.to(args.device)

    cov_matrix = torch.cov(tri_distance.T)
    inv_matrix = torch.inverse(cov_matrix)

    scores = []
    scores_ori = []
    for i, g_i in enumerate(vectorize_nets):
        t = tri_distance[i]
        ma_dis = torch.matmul(torch.matmul(t.unsqueeze(0), inv_matrix), t.unsqueeze(0).T).item()
        ori_dis = torch.matmul(t, t.T).item()
        scores.append(ma_dis)
        scores_ori.append(ori_dis)

    # 良性的分数越小
    # p = 1-args.malicious
    # 直接假设已经知道攻击者的数量，最优的方案
    p_num = num_benign_clients
    topk_ind = np.argpartition(scores, p_num)[:p_num]
    topk_ind_ori = np.argpartition(scores_ori, p_num)[:p_num]

    record_TNR_TPR(args, topk_ind, writer, iter)
    # args.psum += num_clients - p_num
    # args.nsum += p_num
    # fn = 0
    # for i in range(p_num):
    #     if topk_ind[i] < num_malicious_clients:
    #         fn += 1
    #     else:
    #         args.tn += 1
    # args.tp += num_malicious_clients - fn
    # TPR = args.tp / args.psum
    # TNR = args.tn / args.nsum
    # writer.add_scalar("Metric/TPR", TPR, iter)
    # writer.add_scalar("Metric/TNR", TNR, iter)

    # 记录不经过动态归一化操作的原分数作为对比
    fn_ori = 0
    for i in range(p_num):
        if topk_ind_ori[i] < num_malicious_clients:
            fn_ori += 1
        else:
            args.tn_ori += 1
    args.tp_ori += num_malicious_clients - fn_ori
    TPR = args.tp_ori / args.psum
    TNR = args.tn_ori / args.nsum
    writer.add_scalar("Metric/TPR_ori", TPR, iter)
    writer.add_scalar("Metric/TNR_ori", TNR, iter)
    writer.add_scalar("Metric/TP_ori", args.tp_ori / (num_malicious_clients*(iter-args.start_defence)), iter)
    writer.add_scalar("Metric/TN_ori", args.tn_ori / (num_benign_clients*(iter-args.start_defence)), iter)

    global_model = no_defence_balance([update_params[i] for i in topk_ind], global_model)

    return global_model

def get_pca(data):
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    return data

def fl_defender(global_model, local_models, update_params, args, writer, file_name, iter):
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to(args.device)
    last_g = list(global_model.parameters())[-2].detach().reshape(-1)
    m = len(local_models)
    f_grads = [None for i in range(m)]
    for i in range(m):
        f_grads[i] = (last_g - \
                list(local_models[i].parameters())[-2].detach().reshape(-1))
        # f_grads[i] = grad.reshape(-1)

    cos_list = []
    for i in range(len(f_grads)):
        cos_i = []
        for j in range(len(f_grads)):
            cos_ij = 1- cos(f_grads[i],f_grads[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)

    # cs = smp.cosine_similarity(f_grads) - np.eye(m)
    if iter > args.start_attack and iter % 100 == 0:
        file_path = "/root/project/epics/" + file_name
        if not os.path.exists(file_path):    
            os.makedirs(file_path)
        cos_proj = PCA(n_components=2).fit_transform(cos_list)
        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
        color_map = ['r'] * num_malicious_clients + ['b'] * num_benign_clients
        for i in range(num_clients):
            ax.scatter(cos_proj[i, 0], cos_proj[i, 1], c=color_map[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('PCA Axis 1', fontsize=17)
        ax.set_ylabel('PCA Axis 2', fontsize=17)
        plt.savefig(os.path.join(file_path, str(iter)+'.pdf'))
    cs = torch.tensor(get_pca(cos_list), dtype=torch.float32)
    centroid = torch.quantile(cs, 0.5, dim=0)
    scores = []
    for i in range(len(cs)):
        scores.append(cos(cs[i], centroid))

    # TODO:看是否需要加上历史信息
    # 良性的分数越大
    p_num = num_benign_clients
    topk_ind = np.argpartition(scores, -p_num)[-p_num:]
    
    record_TNR_TPR(args, topk_ind, writer, iter)
    # args.psum += num_clients - p_num
    # args.nsum += p_num
    # fn = 0
    # for i in range(len(topk_ind)):
    #     if topk_ind[i] < num_malicious_clients:
    #         fn += 1
    #     else:
    #         args.tn += 1
    # args.tp += num_malicious_clients - fn
    # TPR = args.tp / args.psum
    # TNR = args.tn / args.nsum
    # writer.add_scalar("Metric/TPR", TPR, iter)
    # writer.add_scalar("Metric/TNR", TNR, iter)
    
    w_glob = global_model.state_dict()
    w_glob = no_defence_balance([update_params[i] for i in topk_ind], w_glob)

    return w_glob

def create_cluster_map_from_labels(expected_number_of_labels, clustering_labels):
    """
    Converts a list of labels into a dictionary where each label is the key and
    the values are lists/np arrays of the indices from the samples that received
    the respective label
    :param expected_number_of_labels number of samples whose labels are contained in
    clustering_labels
    :param clustering_labels list containing the labels of each sample
    :return dictionary of clusters
    """
    assert len(clustering_labels) == expected_number_of_labels

    clusters = {}
    for i, cluster in enumerate(clustering_labels):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(i)
    return {index: np.array(cluster) for index, cluster in clusters.items()}

def determine_biggest_cluster(clustering):
    """
    Given a clustering, given as dictionary of the form {cluster_id: [items in cluster]}, the
    function returns the id of the biggest cluster
    """
    biggest_cluster_id = None
    biggest_cluster_size = None
    for cluster_id, cluster in clustering.items():
        size_of_current_cluster = np.array(cluster).shape[0]
        if biggest_cluster_id is None or size_of_current_cluster > biggest_cluster_size:
            biggest_cluster_id = cluster_id
            biggest_cluster_size = size_of_current_cluster
    return biggest_cluster_id

def crowdguard(helper, validate_users_id, args, global_model, all_models, update_params, dataset_train, dict_users, idxs_users, writer, file_name, iter):
    binary_votes = []
    for validator_id in validate_users_id: # global id
        if args.dataset == 'loan':
            validator_listid = helper.participants_list.index(validator_id)
            validator_train_loader = helper.allStateHelperList[validator_listid].get_trainloader()
        else:
            if args.attack == 'edges' and (validator_id < int(args.num_users * args.malicious)):
                validator_train_loader = DataLoader(args.poison_trainloader.dataset, batch_size=args.local_bs, shuffle=True)
            else:
                if args.gau_noise > 0:
                    noise_level = args.gau_noise / (args.num_users - 1) * validator_id
                    dataset, _ = load_data(args.dataset, args.data_dir, dict_users[validator_id], noise_level)
                else:
                    dataset = DatasetSplit(dataset_train, dict_users[validator_id])
                
                validator_train_loader = DataLoader(dataset, batch_size=args.local_bs, shuffle=True)
        
        detected_suspicious_models = CrowdGuardClientValidation.validate_models(global_model,
                                                                                all_models,
                                                                                idxs_users.index(validator_id), # 将global id转换为相对id
                                                                                validator_train_loader,
                                                                                args.device,
                                                                                file_name,
                                                                                args,
                                                                                iter)
        detected_suspicious_models = sorted(detected_suspicious_models) # 相对id
        votes_of_this_client = [] # 当前验证者对所有模型的投票
        # TODO:用户索引的问题，不是所有客户端都参与训练
        for r_id, g_id in enumerate(idxs_users):
            if g_id == validator_id:
                votes_of_this_client.append(1) # VOTE_FOR_BENIGN
            elif r_id in detected_suspicious_models:
                votes_of_this_client.append(0) # VOTE_FOR_POISONED
            else:
                votes_of_this_client.append(1) # VOTE_FOR_BENIGN
        binary_votes.append(votes_of_this_client)
    print(binary_votes)
    ac_e = AgglomerativeClustering(n_clusters=2, distance_threshold=None,
                                       compute_full_tree=True,
                                       affinity="euclidean", memory=None, connectivity=None,
                                       linkage='single',
                                       compute_distances=True).fit(binary_votes)
    ac_e_labels: list = ac_e.labels_.tolist()
    agglomerative_result = create_cluster_map_from_labels(len(idxs_users), ac_e_labels)
    print(f'Agglomerative Clustering: {agglomerative_result}')
    agglomerative_negative_cluster = agglomerative_result[
            determine_biggest_cluster(agglomerative_result)]
    
    db_scan_input_idx_list = agglomerative_negative_cluster
    print(f'DBScan Input: {db_scan_input_idx_list}')
    db_scan_input_list = [binary_votes[vote_id] for vote_id in db_scan_input_idx_list]

    db = DBSCAN(eps=0.5, min_samples=1).fit(db_scan_input_list)
    dbscan_clusters = create_cluster_map_from_labels(len(agglomerative_negative_cluster),
                                                        db.labels_.tolist())
    biggest_dbscan_cluster = dbscan_clusters[determine_biggest_cluster(dbscan_clusters)]
    print(f'DBScan Clustering: {biggest_dbscan_cluster}')

    single_sample_of_biggest_cluster = biggest_dbscan_cluster[0]
    final_voting = db_scan_input_list[single_sample_of_biggest_cluster]
    negatives = [i for i, vote in enumerate(final_voting) if vote == 1] # VOTE_FOR_BENIGN

    print(f'Negatives: {negatives}')

    record_TNR_TPR(args, negatives, writer, iter)

    w_glob = global_model.to(args.device).state_dict()
    w_glob = no_defence_balance([update_params[i] for i in negatives], w_glob)

    return w_glob

def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    cs_max = np.max(cs, axis=1)
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv, cs_max

class FoolsGold:
    def __init__(self, num_users, use_memory=True):
        self.memory = None
        self.wv_history = []
        self.num_users = num_users
        self.use_memory = use_memory
    
    def score_gradients(self, helper, global_model, local_models, idxs_users, update_params, args, writer, iter, file_name):
        if args.dataset == 'loan':
            user_indices = [helper.participants_list.index(user) for user in idxs_users]
            idxs_users = user_indices
        num_clients = max(int(args.frac * args.num_users), 1)
        num_malicious_clients = int(args.malicious * num_clients)
        num_benign_clients = num_clients - num_malicious_clients

        global_model_par = list(global_model.parameters())
        m = len(local_models)
        grads = [None for i in range(m)]
        for i in range(m):
            grads[i]= (global_model_par[-2].cpu().data.numpy() - \
                    list(local_models[i].parameters())[-2].cpu().data.numpy()).reshape(-1)
        grads = np.asarray(grads)
        if self.memory is None:
            self.memory = np.zeros((self.num_users, len(grads[0])))
        
        self.memory[idxs_users]+= grads
        if self.use_memory:
            wv, cs_max = foolsgold(self.memory[idxs_users])
        else:
            wv, cs_max = foolsgold(grads)
        self.wv_history.append(wv)

        scores = wv
        
        # 记录与其他客户端相似度的最大值
        file_path = "/root/project/efiles/" + file_name
        if not os.path.exists(file_path):    
            os.makedirs(file_path)
        with open(os.path.join(file_path, "cs_max.txt"), 'a') as f:
            f.write(', '.join(map(str, cs_max.tolist()))+'\n')
            f.close()

        # 按权重分数为0算TNR、TPR
        benign_client = []
        for i, score in enumerate(scores):
            if score != 0:
                benign_client.append(i)
        record_TNR_TPR(args, benign_client, writer, iter)
        # args.psum += num_clients - len(benign_client)
        # args.nsum += len(benign_client)
        # fn = 0
        # for i in range(len(benign_client)):
        #     if benign_client[i] < num_malicious_clients:
        #         fn += 1
        #     else:
        #         args.tn += 1
        # args.tp += num_malicious_clients - fn
        # TPR = args.tp / args.psum
        # TNR = args.tn / args.nsum
        # writer.add_scalar("Metric/TPR", TPR, iter)
        # writer.add_scalar("Metric/TNR", TNR, iter)

        # 按相对大小算TNR、TPR，分数更大的为良性
        benign_client = np.argpartition(scores, -num_benign_clients)[-num_benign_clients:]
        args.psum_fg += num_clients - len(benign_client)
        args.nsum_fg += len(benign_client)
        fn_fg = 0
        for i in range(len(benign_client)):
            if benign_client[i] < num_malicious_clients:
                fn_fg += 1
            else:
                args.tn_fg += 1
        args.tp_fg += num_malicious_clients - fn_fg
        TPR_fg = args.tp_fg / args.psum_fg
        TNR_fg = args.tn_fg / args.nsum_fg
        writer.add_scalar("Metric/TPR_fg", TPR_fg, iter)
        writer.add_scalar("Metric/TNR_fg", TNR_fg, iter)
        writer.add_scalar("Metric/TP_fg", args.tp_fg / (num_malicious_clients*(iter-args.start_defence)), iter)
        writer.add_scalar("Metric/TN_fg", args.tn_fg / (num_benign_clients*(iter-args.start_defence)), iter)

        w_glob = global_model.state_dict()
        w_glob = no_defence_weight(update_params, w_glob, scores)
        
        return w_glob

def weighted_average_oracle(points, weights, args):
    tot_weights = torch.sum(weights)

    weighted_updates= dict()

    for name, data in points[0].items():
        weighted_updates[name]=  torch.zeros_like(data)
    for w, p in zip(weights, points): # 对每一个agent
        for name, data in weighted_updates.items():
            temp = (w / tot_weights).float().to(args.device)
            temp= temp* (p[name].float())
            # temp = w / tot_weights * p[name]
            if temp.dtype!=data.dtype:
                temp = temp.type_as(data)
            data.add_(temp)

    return weighted_updates

def flshield(helper, args, global_model, update_params, w_locals, idxs_users, iter, writer, writer_file_name, dict_users=None, dataset_train=None):
    w_glob = global_model.state_dict()
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    cos_torch = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to(args.device)

    wv = np.zeros(len(idxs_users), dtype=np.float32)
    grads = [parameters_dict_to_vector_flt(w_local) for w_local in w_locals]
    norms = [torch.norm(grad, p=2).item() for grad in grads]

    # clusters_agg是local_id
    if args.bijective:
        clusters_agg = [[i] for i in range(len(update_params))]
        no_clustering = True
    else:
        clustering_method = args.clustering_methods if args.clustering_methods is not None else 'KMeans'
        _, clusters_agg = cluster_function(grads, clustering_method)
        no_clustering = False
    
    all_validator_evaluations = {}
    evaluations_of_clusters = {}
    count_of_class_for_validator = {}

    for name in idxs_users:  # global_id
        all_validator_evaluations[name] = []

    evaluations_of_clusters[-1] = {}
    for current_id, global_id in enumerate(tqdm(idxs_users, disable=True)):
        val_score_by_class, val_score_by_class_per_example, count_of_class = validation_test(helper, args, global_model, global_id, dict_users, dataset_train)
        val_score_by_class_per_example = [val_score_by_class_per_example[i] for i in range(args.class_num)]
        all_validator_evaluations[global_id] += val_score_by_class_per_example
        evaluations_of_clusters[-1][global_id] = [val_score_by_class[i] for i in range(args.class_num)]
        if global_id not in count_of_class_for_validator.keys():
            count_of_class_for_validator[global_id] = count_of_class

    num_of_clusters = len(clusters_agg)
    
    adj_delta_models = []
    weight_vecs_by_cluster = {}

    for idx, cluster in enumerate(tqdm(clusters_agg, disable=False)):
        evaluations_of_clusters[idx] = {}
        agg_model = copy.deepcopy(global_model)
        weight_vec = np.zeros(len(update_params), dtype=np.float32)

        if len(cluster) != 1:  # cluster representative model
            for i in range(len(update_params)):
                if i in cluster:
                    weight_vec[i] = 1/len(cluster)
        else:
            cos_sims = []
            for local_model_id in range(len(idxs_users)):
                cos_sims.append(cos_torch(grads[local_model_id], grads[clusters_agg[idx][0]]).item())
            cos_sims = np.array(cos_sims)
            # cos_sims = np.array(cosine_similarity(grads, [grads[clusters_agg[idx][0]]])).flatten()
            # logger.info(f'cos_sims by order for client {self.clusters_agg[idx][0]}: {cos_sims}')
            trust_scores = np.zeros(cos_sims.shape)
            for i in range(len(cos_sims)):
                # trust_scores[i] = cos_sims[i]/np.linalg.norm(grads[i])/np.linalg.norm(clean_server_grad)
                trust_scores[i] = cos_sims[i]
                trust_scores[i] = max(trust_scores[i], 0)

            norm_ref = norms[clusters_agg[idx][0]]
            clip_vals = [min(norm_ref/norm, 1) for norm in norms]
            trust_scores = [ts * cv for ts, cv in zip(trust_scores, clip_vals)]
            weight_vec = trust_scores

            contrib_adjustment = 0.25

            # logger.info(f'weight_vec: {weight_vec}')

            # weight_vec = [min(1, elem * contrib_adjustment) for elem in weight_vec]
            weight_vec[idx] = 1
            others_contrib = sum([weight_vec[i] for i in range(len(weight_vec)) if i != idx])
            weight_vec = [elem * contrib_adjustment/others_contrib for elem in weight_vec]
            weight_vec[idx] = 1 - contrib_adjustment
            # logger.info(f'weight_vec: {weight_vec}')
            weight_vec = weight_vec / np.sum(weight_vec)
            others_contrib = sum([weight_vec[i] for i in range(len(weight_vec)) if i != idx])
            num_of_other_contrib = len([weight_vec[i] for i in range(len(weight_vec)) if i != idx and weight_vec[i] > 0])
            # logger.info(f'contribution amount: {others_contrib} from {num_of_other_contrib} clients, own contrib: {weight_vec[idx]}')

            # logger.info(f'weight_vec: {weight_vec}')
        
        weight_vecs_by_cluster[idx] = weight_vec.tolist()

        aggregate_weights = weighted_average_oracle(update_params, torch.tensor(weight_vec), args)
        adj_delta_models.append(aggregate_weights)

        for name, data in agg_model.state_dict().items():
            update_per_layer = aggregate_weights[name]
            try:
                data.add_(update_per_layer)
            except:
                data.add_(update_per_layer.to(data.dtype))
        
        for current_id, global_id in enumerate(tqdm(idxs_users, disable=True)):
            val_score_by_class, val_score_by_class_per_example, count_of_class = validation_test(helper, args, agg_model, global_id, dict_users, dataset_train)
            val_score_by_class_per_example = [val_score_by_class_per_example[i] for i in range(args.class_num)]

            val_score_by_class_per_example = [-val_score_by_class_per_example[i]+all_validator_evaluations[global_id][i] for i in range(args.class_num)]
                
            all_validator_evaluations[global_id]+= val_score_by_class_per_example
            evaluations_of_clusters[idx][global_id] = [-val_score_by_class[i]+evaluations_of_clusters[-1][global_id][i] for i in range(args.class_num)]
    
    # impute 
    # evaluations_of_clusters: representative_models * validator
    evaluations_of_clusters, count_of_class_for_validator = impute_validation(evaluations_of_clusters, count_of_class_for_validator, idxs_users, num_of_clusters, args.class_num, impute_method='iterative')
    
    # record each validator result
    eval_tensor = torch.zeros((len(idxs_users), num_of_clusters, args.class_num)) # validator * representative_models
    for i in range(len(idxs_users)):
        for j in range(num_of_clusters):
            for k in range(args.class_num):
                eval_tensor[i][j][k] = evaluations_of_clusters[j][idxs_users[i]][k]/count_of_class_for_validator[idxs_users[i]][k] if count_of_class_for_validator[idxs_users[i]][k] !=0 else 0
    val_rep_res, _ = torch.min(eval_tensor, dim=2)
    if iter % 30 == 1:
        for val_idx in range(val_rep_res.shape[0]):
            val_res = val_rep_res[val_idx]
            # plot eacj validator's result
            file_path = "/root/project/epics/" + writer_file_name
            if not os.path.exists(file_path):    
                os.makedirs(file_path)
            pic_path = os.path.join(file_path, f'validator{val_idx}_iter{iter}.pdf')
            fig = plt.figure()
            ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
            for i in range(len(val_res)):
                if i < num_malicious_clients:
                    ax.scatter(i, val_res[i], c='r', marker='^')
                else:
                    ax.scatter(i, val_res[i], c='b', marker='o')
            ax.set_xticks([])
            ax.tick_params(axis='y', labelsize=15)
            ax.set_ylabel('min(LIPC)', fontsize=17)
            # 构建图例
            legend_handles = []
            legend_labels = []
            legend_handles.append(plt.Line2D([0], [0], color='r', marker='^', linestyle='None'))
            legend_labels.append(u'Malicious Models')
            legend_handles.append(plt.Line2D([0], [0], color='b', marker='o', linestyle='None'))
            legend_labels.append(u'Benign Models')
            
            # 添加图例到轴
            ax.legend(handles=legend_handles, labels=legend_labels)
            plt.savefig(pic_path)
            plt.close(fig)

    cluster_maliciousness = [len([idx for idx in cluster if idx < num_malicious_clients])/len(cluster) for cluster in clusters_agg]
    validation_container = {
            'evaluations_of_clusters': evaluations_of_clusters,
            'count_of_class_for_validator': count_of_class_for_validator,
            'names': idxs_users,
            'num_of_classes': args.class_num,
            'num_of_clusters': num_of_clusters,
            'all_validator_evaluations': all_validator_evaluations,
            'epoch': iter,
            'params': args,
            'cluster_maliciousness': cluster_maliciousness,
            'adversarial_num': num_malicious_clients,
        }

    valProcessor = ValidationProcessor(validation_container=validation_container)
    wv_by_cluster = valProcessor.run()

    for idx, cluster in enumerate(clusters_agg):
        for cl_id in cluster:
            # wv[cl_id] = wv_by_cluster[idx]
            if no_clustering:
                wv[cl_id] = wv_by_cluster[cl_id]
            else:
                # wv[cl_id] = wv_by_cluster[idx] * len(cluster) * wv[cl_id]
                wv[cl_id] = wv_by_cluster[idx]

    wv = wv/np.sum(wv)

    benign_client = []
    for i, result in enumerate(wv):
        if result > 0:
            benign_client.append(i)
    record_TNR_TPR(args, benign_client, writer, iter)

    w_glob = no_defence_weight(update_params, w_glob, wv)

    return w_glob

def lbfgs(S_k_list, Y_k_list, v, device):  # n=window size, k=parameter size
    v = v.unsqueeze(0)
    curr_S_k = torch.stack(S_k_list, dim=0)  # (n, k)
    curr_Y_k = torch.stack(Y_k_list, dim=0)  

    S_k_time_Y_k = torch.mm(curr_S_k, curr_Y_k.T)  # (n, n)
    S_k_time_S_k = torch.mm(curr_S_k, curr_S_k.T)  

    R_k = torch.triu(S_k_time_Y_k)  # 上三角矩阵 (n, n)
    L_k = S_k_time_Y_k - R_k  # 下三角矩阵 (n, n)

    sigma_k = Y_k_list[-1] @ S_k_list[-1].T / (S_k_list[-1] @ S_k_list[-1].T)  # 标量
    print("fenmu:", S_k_list[-1] @ S_k_list[-1].T)

    D_k_diag = torch.diag(S_k_time_Y_k)  # 对角线元素

    upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)  # 上部分 (n, 2n)
    lower_mat = torch.cat([L_k.T, -torch.diag(D_k_diag)], dim=1)  # 下部分 (n, 2n)
    mat = torch.cat([upper_mat, lower_mat], dim=0)  # 最终拼接 (2n, 2n)

    mat_inv = torch.inverse(mat)  # (2n, 2n)
    print("det:", torch.det(mat))

    approx_prod = sigma_k * v  # 初始近似向量，(1, k)

    p_mat = torch.cat([torch.matmul(curr_S_k, sigma_k * v.T), torch.matmul(curr_Y_k, v.T)], dim=0)  # (2n, 1)
    approx_prod -= torch.mm(
        torch.mm(torch.cat([sigma_k * curr_S_k.T, curr_Y_k.T], dim=1), mat_inv), p_mat
    ).T  # 修正结果转置回 (1, k)

    return approx_prod.squeeze(0)



def fldetector_GapStatistics(score, nobyz):
    nrefs = 10
    ks = range(1, 8)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
    score = (score - min)/(max-min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m]-center[label_pred[m]]) for m in range(len(score))])
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m]-center[label_pred[m]]) for m in range(len(rand))])
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    #print(gapDiff)
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i+1
            break
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        print('Attack Detected!')
        return 1

def fldetector_kmeans(score, nobyz):
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
        #0 is the label of malicious clients
        label_pred = 1 - label_pred
    # real_label=np.ones(100)
    # real_label[:nobyz]=0
    # acc=len(label_pred[label_pred==real_label])/100
    # recall=1-np.sum(label_pred[:nobyz])/nobyz
    # fpr=1-np.sum(label_pred[nobyz:])/(100-nobyz)
    # fnr=np.sum(label_pred[:nobyz])/nobyz
    # print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
    # print(silhouette_score(score.reshape(-1, 1), label_pred))
    return label_pred # benign为1，恶意为0

def fldetector(args, w_glob, w_updates, writer, iter, file_name):
    weight = parameters_dict_to_vector_flt(w_glob)
    param_list = [parameters_dict_to_vector_flt(w_update) for w_update in w_updates]

    if iter - args.start_defence > 3:
        hvp = lbfgs(args.weight_record, args.grad_record, weight - args.last_weight, args.device)
    else:
        hvp = None

    if hvp != None:
        print("hvp:", hvp)
        print("param_list", args.old_grad_list[0])
        pred_grad = []
        distance = []
        for i in range(len(args.old_grad_list)):
            pred_grad.append(args.old_grad_list[i] + hvp)
        
        distance = np.array([torch.norm(pred_grad[i]-param_list[i], p=2).item() for i in range(len(param_list))])
        distance = distance / np.sum(distance)
        args.malicious_score = np.row_stack((args.malicious_score, distance))
    
    label_pred = np.ones(args.num_users)
    if args.malicious_score.shape[0] >= 11:
        if args.start_record_iter == None:
            args.start_record_iter = iter
        # if fldetector_GapStatistics(np.sum(args.malicious_score[-10:], axis=0), args.num_users*args.malicious):
        label_pred = fldetector_kmeans(np.sum(args.malicious_score[-10:], axis=0), args.num_users*args.malicious)
        if iter % 30 == 1:
            consistency_scores = np.sum(args.malicious_score[-10:], axis=0)
            file_path = "/root/project/efiles/" + file_name
            if not os.path.exists(file_path):    
                os.makedirs(file_path)
            with open(os.path.join(file_path, "consistency_scores.txt"), 'a') as f:
                f.write(', '.join(map(str, consistency_scores.tolist()))+'\n')
                f.close()
    
    benign_client = []
    for idx, pred in enumerate(label_pred):
        if pred == 1:
            benign_client.append(idx)
    if args.start_record_iter != None:
        record_TNR_TPR(args, benign_client, writer, iter-(args.start_record_iter-args.start_defence)+1)
    w_glob = no_defence_balance([w_updates[i] for i in benign_client], w_glob)
    new_weight = parameters_dict_to_vector_flt(w_glob)
    grad = new_weight - weight

    if iter - args.start_defence > 1:
        args.weight_record.append(weight-args.last_weight)
        args.grad_record.append(grad-args.last_grad)
    
    if len(args.weight_record) > 10:
        del args.weight_record[0]
        del args.grad_record[0]
    
    args.last_weight = weight
    args.last_grad = grad
    args.old_grad_list = param_list

    return w_glob

def model_to_square_matrix(net_dict):
    net_vector = parameters_dict_to_vector_flt(net_dict)
    n = int(torch.ceil(torch.sqrt(torch.tensor(net_vector.size(0), dtype=torch.float32))))
    square_matrix = torch.zeros((n, n), dtype=net_vector.dtype, device=net_vector.device)
    square_matrix.view(-1)[:net_vector.size(0)] = net_vector

    return square_matrix

from scipy.fftpack import dct
def freqfed(args, w_locals, w_updates, global_model, writer, iter, file_name):
    dct_vectors = []

    for net_dict in w_locals:
        square_matrix = model_to_square_matrix(net_dict)
        square_matrix_np = square_matrix.cpu().numpy()
        dct_matrix_np = dct(dct(square_matrix_np.T, norm='ortho').T, norm='ortho')
        dct_matrix_torch = torch.from_numpy(dct_matrix_np).to(args.device)

        size = dct_matrix_torch.shape[0]
        threshold = size // 2
        low_freq_components = []

        for i in range(threshold):
            for j in range(threshold):
                if i + j <= threshold:
                    low_freq_components.append(dct_matrix_torch[i, j])
        
        dct_vectors.append(torch.tensor(low_freq_components))
    
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to(args.device)
    cos_list=[]
    for i in range(len(dct_vectors)):
        cos_i = []
        for j in range(len(dct_vectors)):
            cos_ij = 1- cos(dct_vectors[i],dct_vectors[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(w_locals)):
            benign_client.append(i)
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    
    record_TNR_TPR(args, benign_client, writer, iter)
    
    if iter > args.start_attack and iter % 100 == 1:
        file_path = "/root/project/epics/" + file_name
        if not os.path.exists(file_path):    
            os.makedirs(file_path)
        cos_proj = PCA(n_components=2).fit_transform(cos_list)
        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
        color_map = ['r'] * num_malicious_clients + ['b'] * num_benign_clients
        marker_map = ['o' if x >= 0 else '^' for x in clusterer.labels_]
        for i in range(num_clients):
            ax.scatter(cos_proj[i, 0], cos_proj[i, 1], c=color_map[i], marker=marker_map[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('PCA Axis 1', fontsize=17)
        ax.set_ylabel('PCA Axis 2', fontsize=17)
        plt.savefig(os.path.join(file_path, str(iter)+'.pdf'))

    global_model = no_defence_balance([w_updates[i] for i in benign_client], global_model)

    return global_model