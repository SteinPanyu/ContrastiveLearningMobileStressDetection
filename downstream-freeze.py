from pathlib import Path
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=0)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_block1 = ConvBlock(1, 32, 8)
        self.conv_block2 = ConvBlock(32, 64, 4)
        self.conv_block3 = ConvBlock(64, 128, 2)
        
        self.max_pooling = nn.MaxPool1d(kernel_size=4, stride=2)
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)
        
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.max_pooling(x)
        
        x = self.conv_block2(x)
        x = self.max_pooling(x)

        x = self.conv_block3(x)
        x = self.global_max_pooling(x)
        
        x = self.flatten(x)

        return x

class DownstreamEncoder(nn.Module):
    def __init__(self):
        super(DownstreamEncoder, self).__init__()
        self.encoder = Encoder()
    
    def forward(self, x):
        x = self.encoder(x)

        return x
    
premodel_state_paths = sorted([sensor for sensor in Path('new_pretrained_models').iterdir()], key=lambda s: s.stem)

premodel_states = []
for premodel_state_path in premodel_state_paths:
    premodel_state = torch.load(premodel_state_path)

    layers_to_remove = [name for name in premodel_state.keys() if 'projection' in name]
    
    for name in layers_to_remove:
        del premodel_state[name]
    
    premodel_states.append(premodel_state)

class DownstreamModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_sensors, premodel_states) -> None:
        super(DownstreamModel, self).__init__()

        self.encoders = nn.ModuleList([DownstreamEncoder() for _ in range(len(premodel_states))])
        for i,encoder in enumerate(self.encoders):
            encoder.load_state_dict(premodel_states[i])
            for params in encoder.parameters():
                params.requires_grad = False
        
        self.batch_norm = nn.BatchNorm1d(num_sensors)

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Linear(num_sensors*embed_dim, 2)
    
    def forward(self, x):
        
        features = [encoder(x[:,i,:].unsqueeze(1)).unsqueeze(1) for i,encoder in enumerate(self.encoders)]
        features = torch.concat(features, dim=1)
        
        features = self.batch_norm(features)
        
        x, _ = self.attention(features, features, features)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
    

class MultimodalDataset(Dataset):
    def __init__(self, path:Path, sensor_names:list[str]):
        self.path = path
        self.sensor_name = sensor_names
        self.n_sensors = len(sensor_names)
        df = pd.read_csv(self.path, index_col='timestamp')
        self.input_sequences = df.loc[:,sensor_names]
        self.labels = df.loc[:, 'label']


    def __getitem__(self, index) -> tuple[torch.Tensor, float]:
        sequence_index = self.input_sequences.index.unique()
        original_sequence = self.input_sequences.loc[sequence_index[index]]
        label = self.labels.loc[sequence_index[index]]
        return torch.Tensor(original_sequence.to_numpy().reshape(self.n_sensors, -1)), label.iloc[0]


    def __len__(self):
        return len(self.input_sequences.index.unique())

batch_size = 128
sensors = sorted(['RRI','HRT'])

participant_labeled_data = [labeled_path for labeled_path in Path('Intermediate/proc/labeled_joined/labeled_updated').iterdir()]

kf = KFold(n_splits=5, shuffle=False)

average_acc = []
average_f1 = []
average_auc = []

for i, (train_index, test_index) in enumerate(kf.split(participant_labeled_data)):
    train_participants = [participant_labeled_data[idx] for idx in train_index]
    test_val_participants = [participant_labeled_data[idx] for idx in test_index]

    test_participants = test_val_participants[:len(test_val_participants)//2]
    val_participants = test_val_participants[len(test_val_participants)//2:]

    model = DownstreamModel(embed_dim=128, num_heads=8, num_sensors=2, premodel_states=premodel_states)
    model = model.cuda()

    train_datasets = [MultimodalDataset(path, sensors) for path in train_participants]
    train_dataloader = [DataLoader(dataset, batch_size, shuffle=True) for dataset in train_datasets]

    val_datasets = [MultimodalDataset(path, sensors) for path in val_participants]
    val_dataloaders = [DataLoader(dataset, batch_size, shuffle=False) for dataset in val_datasets]

    test_datasets = [MultimodalDataset(path, sensors) for path in test_participants]
    test_dataloaders = [DataLoader(dataset, batch_size, shuffle=False) for dataset in test_datasets]

    
    tr_ep_loss = []
    tr_ep_acc = []

    val_ep_loss = []
    val_ep_acc = []


    EPOCHS = 10

    dsoptimizer = torch.optim.SGD(model.parameters(),lr = 0.01, momentum = 0.9)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(dsoptimizer, step_size=1, gamma=0.98, last_epoch=-1, verbose = True)

    loss_fn = nn.CrossEntropyLoss()



    for epoch in range(EPOCHS):
        
        stime = time.time()
        print("=============== Epoch : %3d ==============="%(epoch+1))
        
        loss_sublist = np.array([])
        acc_sublist = np.array([])
        
        #iter_num = 0
        model.train()
        
        dsoptimizer.zero_grad()
        
        for dataloader in train_dataloader:
            for x,y in dataloader:
                x = x.cuda().float()
                y = y.unsqueeze(1).cuda().float()
                y = torch.concat([y, 1.0-y], dim=1)

                z = model(x)
                
                dsoptimizer.zero_grad()
                
                tr_loss = loss_fn(z,y)
                tr_loss.backward()

                preds = torch.argmax(z, dim=1).detach().cpu()
                labels = torch.argmax(y, dim=1).cpu()
                
                dsoptimizer.step()
                
                loss_sublist = np.append(loss_sublist, tr_loss.cpu().data)
                acc_sublist = np.append(acc_sublist,np.array(preds==labels).astype('int'))

        print('ESTIMATING TRAINING METRICS.............')
        
        print('TRAINING BINARY CROSSENTROPY LOSS: ',np.mean(loss_sublist))
        print('TRAINING BINARY ACCURACY: ',np.mean(acc_sublist))
        
        tr_ep_loss.append(np.mean(loss_sublist))
        tr_ep_acc.append(np.mean(acc_sublist))
        
        print('ESTIMATING VALIDATION METRICS.............')
        
        model.eval()
        
        loss_sublist = np.array([])
        acc_sublist = np.array([])
        
        with torch.no_grad():
            for dataloader in val_dataloaders:
                for x,y in dataloader:
                    x = x.cuda().float()
                    y = y.unsqueeze(1).cuda().float()
                    y = torch.concat([y, 1.0-y], dim=1)
                    
                    z = model(x)

                    val_loss = loss_fn(z,y)

                    preds = torch.argmax(z, dim=1).detach().cpu()
                    labels = torch.argmax(y, dim=1).cpu()

                    loss_sublist = np.append(loss_sublist, val_loss.cpu().data)
                    acc_sublist = np.append(acc_sublist,np.array(preds==labels).astype('int'))
        
        print('VALIDATION BINARY CROSSENTROPY LOSS: ',np.mean(loss_sublist))
        print('VALIDATION BINARY ACCURACY: ',np.mean(acc_sublist))
        
        val_ep_loss.append(np.mean(loss_sublist))
        val_ep_acc.append(np.mean(acc_sublist))
        
        lr_scheduler.step()
        
        print('Saving model...')
        torch.save(model.state_dict(), 'downstream_models/downstream_HRT_RRI_freeze_{}.pt'.format(i))
        
        print("Time Taken : %.2f minutes"%((time.time()-stime)/60.0))

    
    # Model Testing
    
    loss_fn = nn.CrossEntropyLoss()

    model = DownstreamModel(128, 8, 2, premodel_states).cuda()
    model.load_state_dict(torch.load('downstream_models/downstream_HRT_RRI_freeze_{}.pt'.format(i)))

    model.eval()
        
    loss_sublist = np.array([])
    acc_sublist = np.array([])
    f1_sublist = np.array([])
    
    with torch.no_grad():
        for dataloader in test_dataloaders:
            for x,y in dataloader:
                x = x.cuda().float()
                y = y.unsqueeze(1).cuda().float()
                y = torch.concat([y, 1.0-y], dim=1)

                z = model(x)

                val_loss = loss_fn(z,y)

                preds = torch.argmax(z, dim=1).detach().cpu()
                labels = torch.argmax(y, dim=1).cpu()

                f1_sublist = np.append(f1_sublist, f1_score(labels, preds))
                loss_sublist = np.append(loss_sublist, val_loss.cpu().data)
                acc_sublist = np.append(acc_sublist,np.array(preds==labels).astype('int'))
                

    batch_f1 = np.mean(f1_sublist)            
    batch_accuracy = np.mean(acc_sublist)
    print('TEST_{}'.format(i))
    print('TEST BINARY CROSSENTROPY LOSS: ',np.mean(loss_sublist))
    print('TEST BINARY ACCURACY: ', batch_accuracy)
    print('TEST F1 SCORE: ', batch_f1)

    average_acc.append(batch_accuracy)
    average_f1.append(batch_f1)

print('TOTAL AVERAGE ACCURACY: ', np.mean(average_acc))
print('TOTAL AVERAGE F1: ', np.mean(average_f1))