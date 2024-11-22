import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

def loan_Mytest(net_g, helper):
    net_g.eval()
    test_loss = 0
    correct = 0
    dataset_size = 0
    for i in range(0, len(helper.allStateHelperList)):
        state_helper = helper.allStateHelperList[i]
        data_iterator = state_helper.get_testloader()
        for batch_id, batch in enumerate(data_iterator):
            data, targets = state_helper.get_batch(data_iterator, batch, evaluation=True)
            dataset_size += len(data)
            output = net_g(data)
            test_loss += F.cross_entropy(output, targets,
                                                        reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
    accuracy = 100.0 * (float(correct) / float(dataset_size))
    test_loss /= dataset_size
    return accuracy, test_loss

def loan_Mytest_poison(net_g, helper, args):
    net_g.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    trigger_names = ['num_tl_120dpd_2m', 'num_tl_90g_dpd_24m', 'pub_rec_bankruptcies', 'pub_rec', 'acc_now_delinq', 'tax_liens']
    trigger_values = [10, 80, 20, 100, 20, 100]
    for i in range(0, len(helper.allStateHelperList)):
        state_helper = helper.allStateHelperList[i]
        data_source = state_helper.get_testloader()
        data_iterator = data_source
        for batch_id, batch in enumerate(data_iterator):

            for index in range(len(batch[0])):
                batch[1][index] = args.attack_label
                for j in range(0, len(trigger_names)):
                    name = trigger_names[j]
                    value = trigger_values[j]
                    batch[0][index][helper.feature_dict[name]] = value
                poison_data_count += 1

            data, targets = state_helper.get_batch(data_source, batch, evaluation=True)
            dataset_size += len(data)
            output = net_g(data)
            total_loss += F.cross_entropy(output, targets, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
        
    acc = 100.0 * (float(correct) / float(poison_data_count))
    total_l = total_loss / poison_data_count

    return acc, total_l