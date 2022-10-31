import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP

from torch.utils.data import Dataset, DataLoader

import numpy as np
import time
import pickle
########## Prepare data ##########

from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold


class ConvDataset(Dataset):

    def __init__(self,maxlen,c_to_i,smiles_list,property_list):
        self.smiles_list = smiles_list
        self.maxlen = maxlen
        self.c_to_i = c_to_i
        sequence_list = self.encode_smiles()
        self.sequence_list = sequence_list
        self.property_list = torch.from_numpy(np.array(property_list)) # y

    def encode_smiles(self):
        smiles_list = self.smiles_list
        c_to_i = self.c_to_i
        maxlen = self.maxlen
        sequence_list = []
        for smiles in smiles_list:
            sequence = []
            for s in smiles:
                sequence.append(c_to_i[s])
            sequence = torch.from_numpy(np.array(sequence))
            sequence_list.append(sequence)
        return sequence_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        sample = dict()
        sample['seq'] = self.sequence_list[idx]
        sample['logP'] = self.property_list[idx]
        return sample


class ConvRegressor(torch.nn.Module):

    def __init__(self, n_channel=128, n_conv_layer = 10, kernel_size=3, n_char=46,maxlen = 64):
        super(ConvRegressor, self).__init__()
        convolution_layers = []
        for i in range(n_conv_layer):
            convolution_layers.append(nn.Conv1d(n_channel,n_channel,kernel_size,1,padding = kernel_size//2))
        self.conv = nn.ModuleList(convolution_layers)
        self.fc = nn.Linear(maxlen * n_channel,1)
        self.embedding = nn.Embedding(n_char, n_channel)
        self.activation = nn.ReLU()

    def forward(self, x):
        retval = x
        retval = self.embedding(retval)
        retval = retval.permute((0,2,1))
        # Pass conv and dropout
        for layer in self.conv:
            retval = layer(retval)
            retval = self.activation(retval)
        ###### Flatten
        retval = retval.view(retval.size(0), -1)
        # retval: B*(FL) -> B * (FL)
        retval = self.fc(retval)
        # Reshape into B x F x L
        return retval


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

# Encoding
def get_c_to_i(smiles_list):
    c_to_i = dict()
    for smiles in smiles_list:
        for letter in smiles:
            if letter not in c_to_i:
                c_to_i[letter] = len(c_to_i)
    return c_to_i

# Padding
def adjust_smiles(smiles_list,maxlen):
    for i in range(len(smiles_list)):
        smiles_list[i] = smiles_list[i].ljust(maxlen,'X')

def encode_smiles(smiles_list,c_to_i):
    seq_list = []
    for smiles in smiles_list:
        seq = list(map(lambda x:c_to_i[x],smiles))
        seq_list.append(seq)
    return seq_list

def calculate_logp(smiles_list):
    logp_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        logp = MolLogP(mol)
        logp_list.append(logp)
    return logp_list

# Prepare data
maxlen = 64
smiles_list = load_data('smiles.txt',maxlen,20000)
logp_list = calculate_logp(smiles_list)
c_to_i = get_c_to_i(smiles_list)
c_to_i['X'] = len(c_to_i)
# Save used c_to_i
f = open('c_to_i.pkl','wb')
pickle.dump(c_to_i,f)
f.close()
f = open('c_to_i.pkl','rb')
c_to_i = pickle.load(f)
adjust_smiles(smiles_list,maxlen)
seq_list = np.array(encode_smiles(smiles_list,c_to_i))

kf = KFold(n_splits=5)
splitted = kf.split(seq_list)
final_train_loss = []
final_test_loss = []
for train_index, val_index in splitted:
    # Make train/test/dataloader
    train_smiles = list(map(lambda x:smiles_list[x],train_index))
    test_smiles = list(map(lambda x:smiles_list[x],val_index))
    train_logp = list(map(lambda x:logp_list[x],train_index))
    test_logp = list(map(lambda x:logp_list[x],val_index))
    train_dataset = ConvDataset(maxlen,c_to_i,train_smiles, train_logp)
    test_dataset = ConvDataset(maxlen,c_to_i,test_smiles, test_logp)

    #Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=1)

    # Set trainig parameter
    model = ConvRegressor(128, 2, 3, len(c_to_i),maxlen)
    lr = 1e-4
    num_epoch = 30
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()


    loss_list = []
    st = time.time()
    print ('train start')
    # Load model
    model = model.cuda()
    train_loss_list = []
    test_loss_list = []
    for epoch in range(num_epoch):
        epoch_loss = []
        for i_batch, batch in enumerate(train_dataloader):
            x = batch['seq'].cuda()
            y = batch['logP'].cuda()
            x = x.long()
            y = y.float()
            pred = model(x)
            pred = pred.squeeze(-1)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_list.append(loss.data.cpu().numpy())
            epoch_loss.append(loss.data.cpu().numpy())
        mean_loss = np.mean(np.array(loss_list))
        train_loss_list.append(mean_loss)
        loss_list = []
        with torch.no_grad():
            for i_batch, batch in enumerate(test_dataloader):
                x = batch['seq'].cuda()
                y = batch['logP'].cuda()
                x = x.long()
                y = y.float()
                pred = model(x)
                pred = pred.squeeze(-1)
                loss = loss_fn(pred, y)
                loss_list.append(loss.data.cpu().numpy())
        mean_loss = np.mean(np.array(loss_list))
        test_loss_list.append(mean_loss)
    final_train_loss.append(train_loss_list)
    final_test_loss.append(test_loss_list)
    print ('done')
final_train_loss = np.array(final_train_loss)
final_test_loss = np.array(final_test_loss)
mean_train_loss = np.mean(final_train_loss,axis=0)
mean_test_loss = np.mean(final_test_loss,axis=0)
minimum = np.min(mean_test_loss)
index = np.where(mean_test_loss<minimum + 0.0001)[0].tolist()[0]
print (mean_train_loss)
print (mean_test_loss)
print (minimum)
print (index)