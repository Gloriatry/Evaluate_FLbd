import numpy as np
import copy
import torch
import sys
sys.path.append('../')
from torch.utils.data import DataLoader, Dataset
from utils.load_data import load_data

import time

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

def validation_test_fun(helper, args, network, given_test_loader=None, is_poisonous=False, adv_index=-1, tqdm_disable=True, num_classes=10):
    device2 = args.device
    network.eval()
    network.to(device2)
    correct = 0
    correct_by_class = {}
    correct_by_class_per_example = {}
    loss_by_class = {}
    loss_by_class_per_example = {}
    count_per_class = {}
    loss = 0.

    # num_classes = config.num_of_classes_dict[helper.params['type']]

    start = time.time()
    times = []

    dataset_classes = {}
    test_loader = given_test_loader
    # TODO: semantic attack
    # if given_test_loader is not None:
    #     validation_dataset = copy.deepcopy(given_test_loader.dataset)
    #     if helper.params['attack_methods'] in [config.ATTACK_SEMANTIC]:
    #         validation_dataset = torch.utils.data.ConcatDataset([validation_dataset, helper.semantic_dataloader_correct.dataset])
    #     if helper.params['attack_methods'] in [config.ATTACK_AOTT] and False:
    #         if not is_poisonous or True:
    #             validation_dataset = torch.utils.data.ConcatDataset([validation_dataset, helper.clean_val_loader.dataset])
    #         else:
    #             validation_dataset = torch.utils.data.ConcatDataset([validation_dataset, helper.poison_trainloader.dataset])
    #     test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=len(validation_dataset))

    times.append(('data prep', time.time()-start))
    start = time.time()

    for c in range(num_classes):
        count_per_class[c] = 0
        loss_by_class[c] = []
        loss_by_class_per_example[c] = 0.
        correct_by_class[c] = []
        correct_by_class_per_example[c] = []
    
    with torch.no_grad():
        for batch_id, (data, targets) in enumerate(test_loader):
            # if helper.params['type'] != config.TYPE_LOAN:
            #     if is_poisonous and helper.params['attack_methods']==config.ATTACK_TLF and False:
            #         data, targets, _ = helper.get_poison_batch_for_targeted_label_flip(batch, 4)
            #     else:
            #         data, targets = helper.get_batch(None, batch)
            # else:
            #     data, targets = helper.allStateHelperList[adv_index].get_batch(test_loader, batch, evaluation=True)
            if args.dataset == 'loan':
                data = data.float()
                targets = targets.long()
            data, targets = data.to(device2), targets.to(device2)
            output = network(data)
            loss_func=torch.nn.CrossEntropyLoss(reduction='none')
            pred = output.data.max(1, keepdim=True)[1]
            correct_array = pred.eq(targets.data.view_as(pred))
            correct += correct_array.sum()
            loss_array = loss_func(output, targets)
            loss += loss_array.sum().item()
            class_indices = {}
            for cl in range(num_classes):
                class_indices[cl] = (targets==cl)
                count_per_class[cl] += (class_indices[cl]).sum().item()

                # loss_by_class[cl] += loss_array[class_indices[cl]].sum().item()
                # correct_by_class[cl] += correct_array[class_indices[cl]].sum().item()     
                loss_by_class[cl] += [loss_val.item() for loss_val in loss_array[class_indices[cl]]]
                correct_by_class[cl] += [correct_val.item() for correct_val in correct_array[class_indices[cl]]]
                # if helper.params['non_complying_validators'] and is_poisonous:
                #     # only keep one example per class
                #     loss_by_class[cl] = loss_by_class[cl][:1]
                #     correct_by_class[cl] = correct_by_class[cl][:1]
                #     count_per_class[cl] = 1

            # data, targets = data.to(config.device), targets.to(config.device)
                                                          
    # network.to(config.device)

    times.append(('forward pass', time.time()-start))
    start = time.time()
                                                        
            
    for class_label in range(num_classes):
        cap_on_per_class = True
        if count_per_class[class_label] > 30 and cap_on_per_class:
            count_per_class[class_label] = 30
            loss_by_class[class_label] = loss_by_class[class_label][:30]
            correct_by_class[class_label] = correct_by_class[class_label][:30]

        loss_by_class[class_label] = np.sum(loss_by_class[class_label])
        correct_by_class[class_label] = np.sum(correct_by_class[class_label])
        
        if count_per_class[class_label] == 0:
            correct_by_class[class_label] = 0
            correct_by_class_per_example[class_label] = np.nan
            loss_by_class[class_label] = 0.
            loss_by_class_per_example[class_label] = np.nan
        else:
            correct_by_class_per_example[class_label] = 100. * correct_by_class[class_label]/ count_per_class[class_label]
            loss_by_class_per_example[class_label] = loss_by_class[class_label]/ count_per_class[class_label]

    validation_metric = 'LIPC'

    times.append(('lipc calculation', time.time()-start))
    start = time.time()

    times = [(name, time/sum([t[1] for t in times])) for name, time in times]
    # print('Validation time breakdown: ', times)

    if validation_metric == 'LIPC':
        return loss_by_class, loss_by_class_per_example, count_per_class
    elif validation_metric == 'loss_impact_only':
        total_loss = np.sum([loss_by_class[cl] for cl in range(num_classes)])
        for cl in range(num_classes):
            loss_by_class[cl] = total_loss/num_classes
            count_per_class[cl] = 1
            loss_by_class_per_example[cl] = total_loss/num_classes
        return loss_by_class, loss_by_class_per_example, count_per_class
    elif validation_metric == 'accuracy':
        return correct_by_class, correct_by_class_per_example, count_per_class

    return 100. * correct / len(test_loader.dataset), loss_by_class, loss_by_class_per_example, count_per_class


def validation_test(helper, args, target_model, validator_id, dict_users=None, dataset_train=None):
    if args.dataset == 'loan':
        validator_id = helper.participants_list.index(validator_id)
        val_test_loader = helper.allStateHelperList[validator_id].get_trainloader()
    else:
        # _, val_test_loader = helper.train_data[validator_idx]
        # val_test_loader = helper.val_data[validator_idx]
        if args.gau_noise > 0:
            noise_level = args.gau_noise / (args.num_users - 1) * validator_id
            dataset, _ = load_data(args.dataset, args.data_dir, dict_users[validator_id], noise_level)
        else:
            dataset = DatasetSplit(dataset_train, dict_users[validator_id])
        # print(dict_users[validator_id])
        val_test_loader = DataLoader(dataset, batch_size=args.local_bs, shuffle=True)
    # if validator_id in helper.adversarial_namelist:
    #     is_poisonous_validator = True
    # else:
    #     is_poisonous_validator = False
    if args.dataset == 'loan':
        # val_acc, val_acc_by_class = helper.validation_test_for_loan(target_model, val_test_loader, is_poisonous_validator, adv_index=0)
        return validation_test_fun(helper, args, target_model, val_test_loader, num_classes=9)
    else:
        return validation_test_fun(helper, args, target_model, val_test_loader, num_classes=10)