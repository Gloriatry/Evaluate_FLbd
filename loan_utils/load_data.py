import os
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from collections import Counter, OrderedDict
import numpy as np

class LoanDataset(data.Dataset):
    # label from 0 ~ 8
    # ['Current', 'Fully Paid', 'Late (31-120 days)', 'In Grace Period', 'Charged Off',
    # 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Fully Paid',
    # 'Does not meet the credit policy. Status:Charged Off']

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.train = True
        self.df = pd.read_csv(csv_file)
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        loans_df = self.df.copy()
        x_feature = list(loans_df.columns)
        x_feature.remove('loan_status')
        x_val = loans_df[x_feature]
        y_val = loans_df['loan_status']
        # x_val.head()
        y_val=y_val.astype('int')
        x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.2, random_state=42)
        self.data_column_name = x_train.columns.values.tolist() # list
        self.label_column_name= x_test.columns.values.tolist()
        self.train_data = x_train.values # numpy array
        self.test_data = x_test.values

        self.train_labels = y_train.values
        self.test_labels = y_test.values

        # print(len(self.train_data[0]))
        print(csv_file, "train", len(self.train_data),"test",len(self.test_data))

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            data, label = self.train_data[index], self.train_labels[index]
        else:
            data, label = self.test_data[index], self.test_labels[index]

        return data, label

    def SetIsTrain(self,isTrain):
        self.train =isTrain

    def getPortion(self,loan_status=0):
        train_count= 0
        test_count=0
        for i in range(0,len(self.train_labels)):
            if self.train_labels[i]==loan_status:
                train_count+=1
        for i in range(0,len(self.test_labels)):
            if self.test_labels[i]==loan_status:
                test_count+=1
        return (train_count+test_count)/ (len(self.train_labels)+len(self.test_labels)), \
               train_count/len(self.train_labels), test_count/len(self.test_labels)

class StateHelper():
    def __init__(self, params):
        self.params= params
        self.name=""

    def load_data(self, filename='./data/loan/loan_IA.csv'):

        self.all_dataset = LoanDataset(filename)

    def get_trainloader(self):

        self.all_dataset.SetIsTrain(True)
        train_loader = torch.utils.data.DataLoader(self.all_dataset, batch_size=self.params.local_bs,
                                                   shuffle=True)

        return train_loader

    def get_testloader(self):

        self.all_dataset.SetIsTrain(False)
        test_loader = torch.utils.data.DataLoader(self.all_dataset,
                                                  batch_size=self.params.bs,
                                                  shuffle=False)

        return test_loader

    def get_poison_trainloader(self):
        self.all_dataset.SetIsTrain(True)

        return torch.utils.data.DataLoader(self.all_dataset,
                                           batch_size=self.params.local_bs,
                                           shuffle=True)

    def get_poison_testloader(self):

        self.all_dataset.SetIsTrain(False)

        return torch.utils.data.DataLoader(self.all_dataset,
                                           batch_size=self.params.bs,
                                           shuffle=False)

    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.float().to(self.params.device)
        target = target.long().to(self.params.device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

class LoanHelper():
    def load_data(self, args):
        self.statehelper_dic ={}
        self.allStateHelperList=[]
        self.participants_list=[]
        # self.adversarial_namelist=['FL', 'IL', 'NJ', 'PA', 'OH', 'GA', 'VA', 'NC', 'MI', 'MD', 'AZ', 'CT', 'MO', 'TN']
        self.adversarial_namelist = []
        self.benign_namelist = []
        self.feature_dict = dict()

        filepath_prefix='../project/data/loan/'
        all_userfilename_list = os.listdir(filepath_prefix)
        for j in range(0,len(all_userfilename_list)):
            user_filename = all_userfilename_list[j]
            state_name = user_filename[5:7]
            self.participants_list.append(state_name)
            helper = StateHelper(params=args)
            file_path = filepath_prefix+ user_filename
            if state_name == ['ZZ']:
                if args.aggregation_methods != 'fltrust':
                    continue
            # if self.params['aggregation_methods'] == config.AGGR_FLTRUST and j == len(all_userfilename_list)-1:
            #     file_path = f'{os.getcwd()}/data/loan_dump/loan_root_copy2.csv'
            helper.load_data(file_path)
            self.allStateHelperList.append(helper)
            helper.name = state_name
            self.statehelper_dic[state_name] = helper
            if j==0:
                for k in range(0,len(helper.all_dataset.data_column_name)):
                    self.feature_dict[helper.all_dataset.data_column_name[k]]=k


        # self.benign_namelist = [x for x in self.participants_list if x not in self.adversarial_namelist]

        # if params_loaded['is_random_namelist']==False:
        #     self.participants_list = params_loaded['participants_namelist']
        # else:
        #     self.participants_list= self.benign_namelist+ self.adversarial_namelist


        # lsrs = []

        # for id in range(len(allStateHelperList)):
        #     stateHelper = allStateHelperList[id]
        #     lsr = get_label_skew_ratios(stateHelper.all_dataset.train_labels, id)
        #     lsrs.append(lsr)



def get_label_skew_ratios(self, y_labels, id, num_of_classes=9):
        dataset_classes = {}
        # for ind, x in enumerate(dataset):
        #     _, label = x
        #     #if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
        #     #    continue
        #     if label in dataset_classes:
        #         dataset_classes[label] += 1
        #     else:
        #         dataset_classes[label] = 1
        # for key in dataset_classes.keys():
        #     # dataset_classes[key] = dataset_classes[key] 

        #     dataset_classes[key] = float("{:.2f}".format(dataset_classes[key]/len(dataset)))

        dataset_dict = OrderedDict(Counter(y_labels))
        for y in range(num_of_classes):
            if y not in dataset_dict.keys():
                dataset_dict[y] = 0
        dataset_dict = OrderedDict(sorted(dataset_dict.items()))
        # for c in range(num_of_classes):
        #     dataset_classes.append(dataset_dict[c])
        # dataset_classes = np.array(dataset_classes)
        # print(dataset_classes)
        dataset_classes = np.array(list(dataset_dict.values()))
        dataset_classes = dataset_classes/np.sum(dataset_classes)
        return dataset_classes