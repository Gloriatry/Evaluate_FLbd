import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np

class CombinedTestDataset(Dataset):
    def __init__(self, loan_datasets, sample_ratio=0.1):
        self.loan_datasets = loan_datasets
        self.sample_ratio = sample_ratio
        self.combined_data = []
        self.combined_labels = []
 
        # Combine and sample data from each loan dataset's test set
        for loan_dataset in loan_datasets:
            indices = np.random.choice(
                len(loan_dataset.test_data),
                size=int(len(loan_dataset.test_data) * sample_ratio),
                replace=False
            )
            sampled_data = loan_dataset.test_data[indices]
            sampled_labels = loan_dataset.test_labels[indices]
            self.combined_data.append(sampled_data)
            self.combined_labels.append(sampled_labels)
 
        # Flatten the lists of numpy arrays into a single numpy array
        self.combined_data = np.concatenate(self.combined_data)
        self.combined_labels = np.concatenate(self.combined_labels)
 
    def __len__(self):
        return len(self.combined_data)
 
    def __getitem__(self, idx):
        return self.combined_data[idx], self.combined_labels[idx]

class loan_LocalSeverUpdate():
    def __init__(self, args, loan_helper):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.loan_helper = loan_helper
        state_datasets = []
        for i in range(0, len(loan_helper.allStateHelperList)):
            state_helper = loan_helper.allStateHelperList[i]
            state_data = state_helper.all_dataset
            state_datasets.append(state_data)
        server_dataset = CombinedTestDataset(state_datasets)
        self.ldr_train = DataLoader(server_dataset, batch_size=args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr_b, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep_b):
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(self.ldr_train):
                data, labels = data.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(data)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net, net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class loan_LocalUpdate():
    def __init__(self, args, state_key, loan_helper):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = loan_helper.statehelper_dic[state_key].get_trainloader()
        self.loan_helper = loan_helper
        self.state_key = state_key

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr_b, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep_b):
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                net.zero_grad()
                data, targets = self.loan_helper.statehelper_dic[self.state_key].get_batch(self.ldr_train, batch,
                                                                                    evaluation=False)
                log_probs = net(data)
                loss = self.loss_func(log_probs, targets)
                loss.backward()
                optimizer.step()
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net, net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class loan_LocalMaliciousUpdate():
    def __init__(self, args, state_key, loan_helper):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = loan_helper.statehelper_dic[state_key].get_poison_trainloader()
        self.loan_helper = loan_helper
        self.state_key = state_key
        self.adversarial_index = -1
        if self.adversarial_index == -1:
            self.trigger_names = ['num_tl_120dpd_2m', 'num_tl_90g_dpd_24m', 'pub_rec_bankruptcies', 'pub_rec', 'acc_now_delinq', 'tax_liens']
            self.trigger_values = [10, 80, 20, 100, 20, 100]
    
    def trigger_data(self, batch):
        for xx in range(len(batch[1])):
            batch[1][xx] = self.args.attack_label
            for yy in range(len(self.trigger_names)):
                trigger_name = self.trigger_names[yy]
                trigger_value = self.trigger_values[yy]
                batch[0][xx][self.loan_helper.feature_dict[trigger_name]] = trigger_value
            if xx > len(batch[1]) * self.args.poison_frac:
                break
        
        return batch

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr_m, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep_m):
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                batch = self.trigger_data(batch)
                data, targets = self.loan_helper.statehelper_dic[self.state_key].get_batch(self.ldr_train, batch, False)
                net.zero_grad()
                log_probs = net(data)
                loss = self.loss_func(log_probs, targets)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net, net.state_dict(), sum(epoch_loss) / len(epoch_loss)