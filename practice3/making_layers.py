import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP

from torch.utils.data import Dataset, DataLoader

import numpy as np
import time
########## Prepare data ##########

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
        length_sequence = [maxlen]
        stride = 1
        dilation = 1
        padding = kernel_size//2
        self.embedding = nn.Embedding(n_char, n_channel)
        self.activation = nn.ReLU()
        for i in range(n_conv_layer):
            l = (length_sequence[-1] + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1
            l = int(l)
            length_sequence.append(l)
        for i in range(n_conv_layer):
            convolution_layers.append(Convolution(n_channel,n_channel,kernel_size,stride,padding,self.activation))
            #convolution_layers.append(SkipConnection(n_channel,kernel_size,stride,padding,self.activation))
            #convolution_layers.append(Inception(n_channel,n_channel,5,1,self.activation))
        self.conv = nn.ModuleList(convolution_layers)
        self.fc = nn.Linear(length_sequence[-1] * n_channel, 1)

    def forward(self, x):
        retval = x
        retval = self.embedding(retval)
        retval = retval.permute((0,2,1))
        # Pass conv and dropout
        for layer in self.conv:
            retval = layer(retval)
            retval = self.activation(retval)
        ###### Flatten           
        retval = torch.reshape(retval,(retval.size(0),-1))
        retval = self.fc(retval)
        return retval

class Convolution(nn.Module):
    
    def __init__(self,n_in,n_out,kernel_size,stride,padding,activation):
        
        super().__init__()
        self.conv = nn.Conv1d(n_in,n_out,kernel_size,stride,padding)
        self.activation = activation

    def forward(self,x):
        retval = self.conv(x)
        retval = self.activation(retval)
        return retval

# New layers
class SkipConnection(nn.Module):
    
    def __init__(self,n_channel,kernel_size,stride,padding,activation):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channel,n_channel,kernel_size,stride,padding)
        self.conv2 = nn.Conv1d(n_channel,n_channel,kernel_size,stride,padding)
        self.activation = activation
    
    def forward(self,x):
        retval = self.conv1(x)
        retval = self.activation(retval)
        retval = self.conv2(retval)
        retval = x + retval
        return self.activation(retval)

class Inception(nn.Module):
    
    def __init__(self,n_in,n_out,max_kernel_size,stride,activation):
        super().__init__()
        maximal_kernel = (max_kernel_size - max_kernel_size//2)*2 + 1
        num_conv = int(maximal_kernel/2) + 1
        self.conv_layers = []
        for i in range(num_conv):
            if i == num_conv-1:
                self.conv_layers.append(nn.Conv1d(n_in,int(n_out/num_conv)+n_out%num_conv,2*i+1,1,padding=i))
            else:
                self.conv_layers.append(nn.Conv1d(n_in,int(n_out/num_conv),2*i+1,1,padding=i))
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.activation = activation

    def forward(self,x):
        output = []
        for layer in self.conv_layers:
            retval = layer(x)
            retval = self.activation(retval)
            output.append(retval)
        output = torch.cat(output,1)

        return output

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

def calculate_logp(smiles_list):
    logp_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        logp = MolLogP(mol)
        logp_list.append(logp)
    return logp_list

# Prepare data
maxlen = 64
num_data = 20000

smiles_list = load_data('smiles.txt',maxlen,num_data)
logp_list = calculate_logp(smiles_list)
adjust_smiles(smiles_list,maxlen)
c_to_i = get_c_to_i(smiles_list)

num_train = 15000

# Make train/test/dataloader
train_smiles = smiles_list[:num_train]
test_smiles = smiles_list[num_train:num_data]
train_logp = logp_list[:num_train]
test_logp = logp_list[num_train:num_data]
train_dataset = ConvDataset(maxlen,c_to_i,train_smiles, train_logp)
test_dataset = ConvDataset(maxlen,c_to_i,test_smiles, test_logp)

#Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=1)

num_layer = 3

# Set trainig parameter
model = ConvRegressor(128, num_layer, 5, len(c_to_i),maxlen)
#print ('state dict: ',model.state_dict)
print (model)
lr = 1e-4
num_epoch = 30
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()


loss_list = []
st = time.time()

# Load model
model = model.cuda()

save_file = 'model.pt'

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
    print (epoch, 'loss: ',np.mean(np.array(loss_list)))
    if epoch%5 == 0 and epoch > 0:
        model_name = 'model_'+str(int(epoch/5))+'.pt'
        torch.save(model.state_dict(),model_name)

end = time.time()
print ('Time:', end-st)

model.eval()
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
print ('test loss: ',np.mean(np.array(loss_list)))


