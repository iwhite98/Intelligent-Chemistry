import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import numpy as np
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DataSet(Dataset):
    
    def __init__(self, smiles_list, fp_list, max_num_atoms):
        self.smiles_list = smiles_list
        self.fp_list = torch.from_numpy(np.array(fp_list))
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
        sample["fp"] = self.fp_list[idx]
        return sample

class ConvRegressor(nn.Module):
    
    def __init__(self, n_channel = 128 , n_conv_layer = 10):
        super().__init__()
        self.activation = nn.ReLU()
        self.embedding = nn.Linear(16, n_channel)
        layer_list = []
        #self.loss_fn = nn.BCELoss()

        for i in range(n_conv_layer):
            layer_list.append(nn.Linear(n_channel, n_channel))

        self.W = nn.ModuleList(layer_list)
        self.fc = nn.Linear(n_channel, 1024)

    def forward(self, x, A):
        retval = x
        retval = self.embedding(retval)
        for w in self.W:
            retval = w(retval)
            retval = torch.matmul(A, retval)
            retval = self.activation(retval)
        retval = retval.mean(1)
        retval = self.fc(retval)
        retval = torch.sigmoid(retval)
        return retval


def load_data(directory, max_num_atoms):
    f = open(directory, "r")
    smiles_list = []
    fp_list = []
    count = 0
    while True:
        smiles = f.readline()
        if not smiles:
            break
        mol = Chem.MolFromSmiles(smiles)
        if mol.GetNumAtoms() > max_num_atoms:
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius = 2, nBits = 1024)
        smiles_list.append(smiles)
        fp_list.append(fp)
        count += 1

    f.close()
    return smiles_list, fp_list

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
        if(dif * 3 < abs(loss_list[epoch]-loss_list[epoch-1])):
            result = True
   
    if result:
        lr = lr * 0.95
        for g in optimizer.param_groups:
            g['lr'] = lr
    
    return lr


def positive_loss_fn(pred, y):
    c = 0.04
    pred = torch.where(pred < 1e-5, pred + 1e-5, pred)
    pred = torch.where(pred > (1-1e-5), pred - 1e-5, pred)
    loss = -(y.mul(torch.log(pred)) + (1 - y).mul(torch.log(1-pred)))
    positive_loss = loss.mul(y)
    negative_loss = loss.mul(1-y)
    total_loss = positive_loss + negative_loss * c
    return total_loss.mean()
    

n_conv_layers = 3
max_num_atoms = 64
smiles_list, fp_list = load_data("fp_smiles.txt", max_num_atoms)
train_smiles = smiles_list[:1000000]
test_smiles = smiles_list[1000000:1001000]
train_fp = fp_list[:1000000]
test_fp = fp_list[1000000:1001000]
plt.hist(np.sum(np.array(train_fp), axis = 1))
plt.savefig('hist3.png')
train_dataset = DataSet(train_smiles, train_fp, max_num_atoms)
test_dataset = DataSet(test_smiles, test_fp, max_num_atoms)

train_dataloader = DataLoader(train_dataset, batch_size = 32,num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size = 32,num_workers=4)

model = ConvRegressor(n_channel = 128, n_conv_layer = 3)
lr = 1e-4
num_epoch = 100
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
loss_fn = nn.BCELoss()

model = model.cuda()
loss_list = []
save_name = 'GCNtoFP.pt'
print('train start !')
for epoch in range(num_epoch):
    epoch_loss = np.empty(0,float)
    fp_ones = np.empty(0,float)
    for i_batch, batch in enumerate(train_dataloader):        
        x = batch['feature'].cuda().float()
        y = batch['fp'].cuda().float()
        adj = batch['adj'].cuda().float()
        pred = model(x, adj)
        #loss = positive_loss_fn(pred,y)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss = np.append(epoch_loss, loss.data.cpu())
    print(epoch, 'loss : ', np.mean(epoch_loss))
    loss_list.append(np.mean(epoch_loss))
    lr = reduce_lr(loss_list, epoch, lr, optimizer)
#    if epoch == num_epoch - 1 :
#        torch.save(model.state_dict(), save_name)


model.eval()
loss_list = np.empty(0, float)
total_pred_list = np.empty((0, 1024), float)
total_true_list = np.empty((0, 1024), float)
with torch.no_grad():
    for i_batch, batch in enumerate(test_dataloader):
        x = batch['feature'].cuda().float()
        y = batch['fp'].cuda().float()
        adj = batch['adj'].cuda().float()
        pred = model(x, adj)
        loss = positive_loss_fn(pred, y)
#        loss_list = np.append(loss_list, loss.data.cpu())
        loss = loss_fn(pred, y)
        total_pred_list = np.append(total_pred_list, pred.data.cpu(), axis = 0)
        total_true_list = np.append(total_true_list, y.data.cpu(), axis = 0)
print('loss : ', np.mean(loss_list))

positive_acc = []
negative_acc = []
total_acc = []
total_pred_list = np.where(total_pred_list > 0.5, 1, 0)
for i in range(len(total_pred_list)):
    positive_count = 0
    negative_count = 0
    num_one = np.sum(total_true_list[i])
    for j in range(1024):
        pred_ele = total_pred_list[i][j]
        true_ele = total_true_list[i][j]
        if pred_ele == 1 and true_ele == 1:
            positive_count += 1

        elif pred_ele == 0 and true_ele == 0:
            negative_count += 1
    
    positive_acc.append(positive_count / num_one)
    negative_acc.append(negative_count / (1024-num_one))
    total_acc.append( (positive_count + negative_count) / 1024)

fig = plt.figure()

ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3) 

ax1.hist(positive_acc)
ax2.hist(negative_acc)
ax3.hist(total_acc)

plt.savefig('hist2.png')









