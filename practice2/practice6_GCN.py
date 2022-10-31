import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP

#Dataset
from torch.utils.data import Dataset, DataLoader
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import numpy as np
import time

######### Prepare data ##########

class GCNDataset(Dataset):

    def __init__(self, max_num_atoms, smiles_list):
        self.max_num_atoms = max_num_atoms
        self.smiles_list = smiles_list
        self.property_list = []
        self.input_feature_list = []
        self.adj_list = []
        self.process_data()
        self.property_list = torch.from_numpy(np.array(self.property_list))

    def process_data(self):
        max_num_atoms = self.max_num_atoms
        for smiles in self.smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            num_atoms = mol.GetNumAtoms()
            # Get padded adj
            adj = GetAdjacencyMatrix(mol) + np.eye(num_atoms)
            padded_adj = np.zeros((max_num_atoms,max_num_atoms))
            padded_adj[:num_atoms,:num_atoms] = adj
            # Get padded feature list
            padded_feature = np.zeros((max_num_atoms,16))
            feature = []
            for i in range(num_atoms):
                feature.append(self.get_atom_feature(mol,i))
            feature = np.array(feature)
            # Get property list
            padded_feature[:num_atoms,:16] = feature
            logp = MolLogP(mol)
            self.property_list.append(logp)
            self.input_feature_list.append(torch.from_numpy(padded_feature))
            self.adj_list.append(torch.from_numpy(padded_adj))

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        sample = dict()
        sample['feature'] = self.input_feature_list[idx]
        sample['adj'] = self.adj_list[idx]
        sample['logp'] = self.property_list[idx]
        return sample


    def one_of_k_encoding(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def get_atom_feature(self, m, atom_i):

        atom = m.GetAtomWithIdx(atom_i)
        return np.array(self.one_of_k_encoding(atom.GetSymbol(),['C','N','O','F','ELSE']) + 
                        self.one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 'ELSE']) +
                        self.one_of_k_encoding(atom.GetFormalCharge(),[-1,0,1,'ELSE']) + 
                        [atom.GetIsAromatic()])    # (5, 6, 4, 1) --> total 16


class GConvRegressor(torch.nn.Module):
    
    def __init__(self, n_feature=128, n_layer = 10):
        super(GConvRegressor, self).__init__()
        layer_list = []
        for i in range(n_layer):
            layer_list.append(nn.Linear(n_feature,n_feature))
        self.W = nn.ModuleList(layer_list)
        self.embedding = nn.Linear(16, n_feature) # Do not use Embedding!
        self.fc = nn.Linear(n_feature, 1)

    def forward(self, x, A): 
        x = self.embedding(x)
        for l in self.W:
            x = l(x)
            #x = torch.einsum('ijk,ikl->ijl', (A.clone(), x))
            #x = torch.matmul(A,x)
            #x = F.relu(x)
        x = x.mean(1)

        retval = self.fc(x)

        return retval

class SkipConnection(torch.nn.Module):
    
    def __init__(self,n_feature):
        super().__init__()
        self.conv = nn.Linear(n_feature,n_feature)
        self.activation = nn.ReLU()
    
    def forward(self,x,A):
        # AHW
        retval = x
        retval = self.conv(retval)
        retval = torch.matmul(A,retval)
        retval = self.activation(retval)
        retval = self.conv(retval)
        retval = torch.matmul(A,retval)
        retval = self.activation(retval)
        return x + retval

####### Loading data ########
def load_data(directory,maxlen,num):
    f = open(directory,'r')
    smiles_list = []
    cnt = 0
    while cnt < num:
        line = f.readline()
        words = line.strip().split('\t')
        smiles = words[-1]
        if len(smiles) <= maxlen-1:
            cnt += 1
            smiles_list.append(words[-1])
    return smiles_list

class GraphConv(nn.Module):
    
    def __init__(self,n_in,n_out,activation):
        self.w = nn.Linaer(n_in,n_out)
        self.activation = activation

    def foward(self,x,A):
        x = self.w(x)
        x = torch.matmul(A,x)
        x = F.relu(x)
        return x

# Prepare data
max_natoms = 64
maxlen = 64
smiles_list = load_data('smiles.txt',maxlen,20000)

# Make train/test/dataloader
train_smiles = smiles_list[:19000]
test_smiles = smiles_list[19001:20000]
train_dataset = GCNDataset(max_natoms,train_smiles)
test_dataset = GCNDataset(max_natoms,test_smiles)

#Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=128, num_workers=1)

# Make model
model = GConvRegressor(128, 5).cuda()

# Training parameter
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
num_epoch = 1000
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()


loss_list = []
st = time.time()

model.cuda()

for epoch in range(num_epoch):
    loss_list = []
    for i_batch, batch in enumerate(train_dataloader):
        x = batch['feature'].cuda().float()
        y = batch['logp'].cuda().float()
        adj = batch['adj'].cuda().float()
        pred = model(x, adj).squeeze(-1)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data.cpu().numpy())
    print (epoch + 1, np.mean(np.array(loss_list)))

end = time.time()
print ('Time:', end-st)
model.eval()
with torch.no_grad():
    loss_list = []
    for i_batch, batch in enumerate(test_dataloader):
        x = batch['feature'].cuda().float()
        y = batch['logp'].cuda().float()
        adj = batch['adj'].cuda().float()
        pred = model(x, adj).squeeze(-1)
        loss = loss_fn(pred, y)
        loss_list.append(loss.data.cpu().numpy())
    print ('validity: ',np.mean(np.array(loss_list)))

