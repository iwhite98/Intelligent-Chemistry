import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold

import random
import pickle
import os
import time
import re
import csv

class DataSet(Dataset):
    
    def __init__(self, smiles_list, Lipo_list, max_num_atoms):
        self.smiles_list = smiles_list
        self.Lipo_list = torch.from_numpy(np.array(Lipo_list))
        self.max_num_atoms = max_num_atoms
        self.feature_list = []
        self.adj_list = []
        self.process_data()

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))
    
    def get_atom_feature(self, m, atom_i):
        atom = m.GetAtomWithIdx(atom_i)
        atom_feature = np.array(self.one_of_k_encoding(atom.GetSymbol(),['C', 'N', 'O', 'F', 'ELSE'])  + self.one_of_k_encoding(atom.GetFormalCharge(), [-1, 0, 1, 'ELSE']) + self.one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 'ELSE']) + [atom.GetIsAromatic()])
        return atom_feature 

    def process_data(self):
        max_num_atoms = self.max_num_atoms
        for smiles in self.smiles_list:
            m = Chem.MolFromSmiles(smiles)
            num_atoms = m.GetNumAtoms()
            atom_feature = np.empty((0,16), int)
            adj = GetAdjacencyMatrix(m) + np.eye(num_atoms)
            
            padded_feature = np.zeros((max_num_atoms, 16))
            padded_adj = np.zeros((max_num_atoms, max_num_atoms))
            
            for i in range(num_atoms):
                atom_feature = np.append(atom_feature, [self.get_atom_feature(m, i)], axis = 0)
            
            padded_feature[:num_atoms, :16] = atom_feature
            padded_adj[:num_atoms, :num_atoms] = adj

            self.feature_list.append(torch.from_numpy(padded_feature))
            self.adj_list.append(torch.from_numpy(padded_adj))

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        sample = dict()
        sample["feature"] = self.feature_list[idx]
        sample["adj"] = self.adj_list[idx]
        sample["Lipo"] = self.Lipo_list[idx]
        return sample
            
class Transfer(nn.Module):
    
    def __init__(self, n_channel = 128 , n_conv_layer1 = 10, n_conv_layer2 = 10):
        super().__init__()
        self.activation = nn.ReLU()
        self.embedding1 = nn.Linear(16, n_channel)
        self.embedding2 = nn.Linear(16, n_channel)
        layer_list1 = []
        layer_list2 = []

        for i in range(n_conv_layer1):
            layer_list1.append(nn.Linear(n_channel, n_channel))
        
        for i in range(n_conv_layer2):
            layer_list2.append(nn.Linear(n_channel, n_channel))
        
        self.W1 = nn.ModuleList(layer_list1)
        self.W2 = nn.ModuleList(layer_list2)
        
        self.linear_fc1 = nn.Linear(n_channel, 1)        
        self.linear_fc2 = nn.Linear(n_channel, 1)
        self.fc_final = nn.Linear(n_channel * 2, 1)

    def load_model1(self, pretrain_dict1):
        rename_dict1 = {key.replace('.', '1.', 1): v for key, v in pretrain_dict1.items()}
        model_dict = self.state_dict()
        load_dict = dict()
        for key in model_dict.keys():
            if key in rename_dict1.keys():
                load_dict[key] = rename_dict1[key]
            else:
                load_dict[key] = model_dict[key]
        self.load_state_dict(load_dict, strict=False)

    def load_model2(self, pretrain_dict1, pretrain_dict2):
        rename_dict1 = {key.replace('.', '1.', 1): v for key, v in pretrain_dict1.items()}
        rename_dict2 = {key.replace('.', '2.', 1): v for key, v in pretrain_dict2.items()}
        model_dict = self.state_dict()
        load_dict = dict()
        for key in model_dict.keys():
            if key in rename_dict1.keys():
                load_dict[key] = rename_dict1[key]
            elif key in rename_dict2.keys():
                load_dict[key] = rename_dict2[key]
            else:
                load_dict[key] = model_dict[key]
        self.load_state_dict(load_dict, strict=True)

    def freeze(self):
        parameters = self.W1.parameters()
        for param in parameters:
            param.requires_grad = False
    
    def concat_fc_layer(self, weight_zero = False):
        w1 = self.linear_fc1.weight
        w2 = self.linear_fc2.weight
        w = torch.cat([w1, w2], dim=1)
        if weight_zero:
            w[:, :128] = 0
        self.fc_final.weight = nn.Parameter(w)

    def forward(self, x, A):
        retval1 = x
        retval1 = self.embedding1(retval1)
        for w in self.W1:
            retval1 = w(retval1)
            retval1 = torch.matmul(A, retval1)
            retval1 = self.activation(retval1)

        retval2 = x
        retval2 = self.embedding2(retval2)
        for w in self.W2:
            retval2 = w(retval2)
            retval2 = torch.matmul(A, retval2)
            retval2 = self.activation(retval2)

        retval = torch.cat([retval1, retval2], dim = 2 )
        retval = retval.mean(1)
        retval = self.fc_final(retval)
        return retval


def load_data(filename = 'Lipophilicity.csv', max_num_atoms = 64):
    
    smiles_list = []
    Lipo_list = []
    
    f = open(filename, 'r')
    data = csv.reader(f)
    next(data)
    for line in data:
        smiles = line[-1]
        Lipo = float(line[-2])
        mol = Chem.MolFromSmiles(smiles)
        if str(type(mol)) == "<class 'NoneType'>":
            continue
        if mol.GetNumAtoms() > max_num_atoms:
            continue
        else:
            smiles_list.append(smiles)
            Lipo_list.append(Lipo)
                   
    return (smiles_list, Lipo_list)

def get_max_num(smiles_list):
    max_num = 0
    for smiles in smiles_list:
        m = Chem.MolFromSmiles(smiles)
        num = m.GetNumAtoms()
        if(num > max_num):
            max_num = num
    return max_num

def reduce_lr(loss_list, epoch, lr, optimizer):
    result = False
    
    if epoch != 0:
        if (loss_list[epoch] >= loss_list[epoch-1]):
            result = True

    if epoch >= 5:
        dif = 0
        for i in range(1,5):
            dif += abs(loss_list[epoch - i]-loss_list[epoch-i-1])
        dif = dif/4
        if(dif*5 < abs(loss_list[epoch]-loss_list[epoch-1])):
            result = True
   
    if result:
        lr = lr * 0.95
        for g in optimizer.param_groups:
            g['lr'] = lr
    
    return lr

n_conv_layers = 3
max_num_atoms = 64
smiles_list, Lipo_list = load_data()

num_data = len(smiles_list) ##642
num_test_data = int(num_data * 0.8)
train_smiles = smiles_list[:num_test_data]
test_smiles = smiles_list[num_test_data : num_data]
train_Lipo = Lipo_list[:num_test_data]
test_Lipo = Lipo_list[num_test_data : num_data]


train_dataset = DataSet(train_smiles, train_Lipo, max_num_atoms )
test_dataset = DataSet(test_smiles, test_Lipo, max_num_atoms) 

train_dataloader = DataLoader(train_dataset, batch_size = 32)
test_dataloader = DataLoader(test_dataset, batch_size = 32)

pretrain_dict1 = torch.load('GCNtoFP.pt')
#pretrain_dict1 = torch.load('qm9_lumo_GCN.pt')
pretrain_dict2 = torch.load('Lipo_GCN.pt')

lr = 1e-3
num_epoch = 2500

loss_fn = nn.MSELoss()
model = Transfer(n_channel = 128, n_conv_layer1 = 3, n_conv_layer2 = 3) 
model.load_model2(pretrain_dict1, pretrain_dict2)
model.freeze()
model.concat_fc_layer(weight_zero = True)
optimizer = optim.Adam(model.parameters(), lr = lr)
model = model.cuda()

loss_list = []
for epoch in range(num_epoch):
    epoch_loss = np.empty(0,float)
    for i_batch, batch in enumerate(train_dataloader):
        
        x = batch['feature'].cuda().float()
        y = batch['Lipo'].cuda().float()
        adj = batch['adj'].cuda().float()
        pred = model(x, adj)
        pred = pred.squeeze(-1)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss = np.append(epoch_loss, loss.data.cpu())

    print(epoch, ' loss : ',np.mean(epoch_loss))
    loss_list.append(np.mean(epoch_loss))
    lr = reduce_lr(loss_list, epoch, lr, optimizer)

f = open('fp_Lipo_zero', 'wb')
pickle.dump(loss_list, f)
f.close()

model.eval()
loss_list = np.empty(0, float)
with torch.no_grad():
    for i_batch, batch in enumerate(test_dataloader):
        
        x = batch['feature'].cuda().float()
        y = batch['Lipo'].cuda().float()
        adj = batch['adj'].cuda().float()
        pred = model(x, adj)
        pred = pred.squeeze(-1)
        loss = loss_fn(pred, y)
        loss_list = np.append(loss_list, loss.data.cpu())

print('--------Lipo GCN transfer zero weight---------')
print('pre-train1 : GCN to fp')
#print('pre-train2 : Lipo')
print('test loss : ', np.mean(loss_list))
