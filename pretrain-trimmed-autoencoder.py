from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse

class UnimodalDataset(Dataset):
    def __init__(self, directory_path: Path, sensor_name: str):
        self.sensor_name = sensor_name
        all_data = []

        for file in directory_path.glob('*_unlabeled.csv'):
            pcode = file.stem.split('_')[0]
            df = pd.read_csv(file)
            df['pcode'] = pcode
            all_data.append(df)

        full_df = pd.concat(all_data, ignore_index=True)
        self.grouped_data = full_df.groupby(['pcode', 'timestamp'])[sensor_name].apply(list).reset_index(name='sequence')
        self.sequences = self.grouped_data['sequence'].tolist()

    def __getitem__(self, index):
        sequence = self.sequences[index]
        return torch.Tensor(sequence).view(1, -1), torch.Tensor(sequence).view(1, -1)

    def __len__(self):
        return len(self.sequences)

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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Reverse operations of the encoder
        # Assume the output of encoder has size [batch, 128, 1]
        self.conv_transpose1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2)  # Output size: [batch, 64, 2]
        self.relu1 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)  # Output size: [batch, 32, 4]
        self.relu2 = nn.ReLU()
        self.conv_transpose3 = nn.ConvTranspose1d(32, 1, kernel_size=16, stride=2, padding=7)  # Output size: [batch, 1, 8]
        self.relu3 = nn.ReLU()
        self.upsample = nn.Upsample(size=60)  # Upsample to match the original size

    def forward(self, x):
        x = x.view(x.size(0), 128, 1)  # Reshape to match decoder input
        x = self.relu1(self.conv_transpose1(x))
        x = self.relu2(self.conv_transpose2(x))
        x = self.relu3(self.conv_transpose3(x))
        x = self.upsample(x)  # Upsample to the original size
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def save_model(model, sensor_name):
    encoder_model = model.encoder
    state_dict = encoder_model.state_dict()

    # Add 'encoder.' prefix to each key in state_dict
    state_dict_with_prefix = {"encoder." + key: value for key, value in state_dict.items()}

    out = Path(f'new_pretrained_models/{sensor_name}.pt')
    out.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    torch.save(state_dict_with_prefix, out)

# ... rest of your


def pretrain_autoencoder(dataloader, model, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        for data in dataloader:
            optimizer.zero_grad()
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
    save_model(model, sensor_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = AutoEncoder().to(device)
pretrain_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
pretrain_criterion = nn.MSELoss()

parser = argparse.ArgumentParser()
parser.add_argument("--sensor_name", type=str, required=True)
args = parser.parse_args()
batch_size = 128

# Load Data and Train
sensor_name = args.sensor_name
directory_path = Path('Intermediate/unlabeled_joined')  # Replace with the actual directory path
participant_dataset = UnimodalDataset(directory_path, sensor_name)
participant_dataloader = DataLoader(participant_dataset, batch_size=batch_size, shuffle=True)

pretrain_autoencoder(participant_dataloader, autoencoder, pretrain_optimizer, pretrain_criterion)
