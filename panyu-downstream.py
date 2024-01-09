from pathlib import Path
import time
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from Funcs.Utility import *
from sklearn.utils.class_weight import compute_class_weight

#Set random seed for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


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

# Define ConvBlock and Encoder classes as before...

class DownstreamEncoder(nn.Module):
    def __init__(self):
        super(DownstreamEncoder, self).__init__()
        self.encoder = Encoder()
    
    def forward(self, x):
        x = self.encoder(x)
        return x

#Load pretrained model states as before...
premodel_state_paths = sorted([sensor for sensor in Path('new_pretrained_models').iterdir()], key=lambda s: s.stem)

premodel_states = []
for premodel_state_path in premodel_state_paths:
    premodel_state = torch.load(premodel_state_path)

    layers_to_remove = [name for name in premodel_state.keys() if 'projection' in name]
    
    for name in layers_to_remove:
        del premodel_state[name]
    
    premodel_states.append(premodel_state)

#Original Downstream Model
class DownstreamModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_sensors, premodel_states):
        super(DownstreamModel, self).__init__()

 #       num_encoders = len(premodel_states)
        num_encoders = num_sensors
        self.encoders = nn.ModuleList([DownstreamEncoder() for _ in range(num_encoders)])

        # Load pretrained model states
        for i, encoder in enumerate(self.encoders):
            encoder.load_state_dict(premodel_states[i])

            # # Freeze the parameters in the encoder
            # for param in encoder.parameters():
            #     param.requires_grad = False

        self.batch_norm = nn.BatchNorm1d(num_sensors)

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.classifier = nn.Linear(num_sensors * embed_dim, 2)
    
    def forward(self, x):

        features = [encoder(x[:, i, :].unsqueeze(1)).unsqueeze(1) for i, encoder in enumerate(self.encoders)]
        features = torch.cat(features, dim=1)
        
        features = self.batch_norm(features)
        
        x, _ = self.attention(features, features, features)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


# #Remove concat and self-attention
# class DownstreamModel(nn.Module):
#     def __init__(self, embed_dim, num_heads, num_sensors, premodel_states):
#         super(DownstreamModel, self).__init__()

#  #       num_encoders = len(premodel_states)
#         num_encoders = num_sensors
#         self.encoders = nn.ModuleList([DownstreamEncoder() for _ in range(num_encoders)])

#         # #Load pretrained model states
#         # for i, encoder in enumerate(self.encoders):
#         #     encoder.load_state_dict(premodel_states[i])

#             # # Freeze the parameters in the encoder
#             # for param in encoder.parameters():
#             #     param.requires_grad = False

#         # Layer to combine features from all sensors
#         self.combining_layer = nn.Linear(num_sensors * embed_dim, embed_dim)  # Adjust dimensions as needed

#         # Batch normalization layer
#         self.batch_norm = nn.BatchNorm1d(embed_dim)

#         self.classifier = nn.Linear(embed_dim, 2)  # Binary classification
    
#     def forward(self, x):

#         encoded_features = [encoder(x[:, i, :].unsqueeze(1)) for i, encoder in enumerate(self.encoders)]

#         # Combine the features from each sensor
#         combined = torch.cat(encoded_features, dim=1)  # Concatenate along the feature dimension
#         combined = self.combining_layer(combined.view(combined.size(0), -1))  # Flatten and pass through the combining layer

#         # Apply batch normalization
#         combined = self.batch_norm(combined)

#         x = self.classifier(combined)

#         return x

# #Attention fusion for different channels
# ##################################################
# class FusionAttention(nn.Module):
#     def __init__(self, num_sensors, embed_dim):
#         super(FusionAttention, self).__init__()
#         self.num_sensors = num_sensors
#         self.query_vectors = nn.Parameter(torch.randn(num_sensors, embed_dim))

#     def forward(self, encoded_features):
#         # encoded_features is a list of tensors, each of shape [batch_size, embed_dim]
        
#         # Stack encoded features to shape [num_sensors, batch_size, embed_dim]
#         stacked_features = torch.stack(encoded_features)

#         # Expand query vectors to shape [num_sensors, batch_size, embed_dim]
#         expanded_queries = self.query_vectors.unsqueeze(1).expand(-1, stacked_features.size(1), -1)

#         # Compute attention scores using dot product, resulting shape [num_sensors, batch_size]
#         attention_scores = torch.einsum('nbd,nbd->nb', expanded_queries, stacked_features)

#         # Apply softmax to get attention weights, shape remains [num_sensors, batch_size]
#         attention_weights = F.softmax(attention_scores, dim=0)

#         # Weighted sum of features, result shape [batch_size, embed_dim]
#         fused_features = torch.einsum('nb,nbd->bd', attention_weights, stacked_features)

#         return fused_features

# class DownstreamModel(nn.Module):
#     def __init__(self, num_sensors, embed_dim, premodel_states):
#         super(DownstreamModel, self).__init__()
#         self.encoders = nn.ModuleList([DownstreamEncoder() for _ in range(num_sensors)])
#         self.fusion_attention = FusionAttention(num_sensors, embed_dim)
#         self.classifier = nn.Linear(embed_dim, 2)  # Adjust the output size as needed

#     def forward(self, x):
#         # Process each sensor's data through its respective encoder
#         encoded_features = [encoder(x[:, i, :].unsqueeze(1)) for i, encoder in enumerate(self.encoders)]
        
#         # Fuse features using attention mechanism
#         fused_features = self.fusion_attention(encoded_features)
        
#         # Further processing (e.g., classification)
#         x = self.classifier(fused_features)
#         return x
# ##################################################
    


class MultimodalDataset(Dataset):
    def __init__(self, file_paths, sensor_names, include_only_real_labels=False):
        all_data = []
        for file in file_paths:
            df = pd.read_csv(file) 
            df['pcode'] = file.stem.split('_')[0]  
            all_data.append(df)

        full_df = pd.concat(all_data, ignore_index=True)
        if include_only_real_labels:
            # Remove rows with pseudo labels
            full_df = full_df[full_df['label_type'] == 'real']
            print("Removed pseudo labels from dataset")
        grouped = full_df.groupby(['pcode', 'timestamp'])

        self.n_sensors = len(sensor_names)

        self.sequences = []
        self.labels = []
        for _, group in grouped:
            self.sequences.append(group[sensor_names].values)  
            self.labels.append(group['label'].iloc[0])

        # Call the method to print label distribution
        self.print_label_distribution()


    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]

        return torch.Tensor(sequence).reshape(self.n_sensors, -1), label

    def __len__(self):
        return len(self.sequences)

    def apply_smote(self):
        # Assuming each sequence in self.sequences is a 2D array where rows are timesteps
        # Flatten each sequence into a 1D array
        data = np.array([s.flatten() for s in self.sequences])
        labels = np.array(self.labels)

        # Apply SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        data_resampled, labels_resampled = smote.fit_resample(data, labels)

        # Reshape data back to original shape and update sequences and labels
        timestep, sensor_dim = self.sequences[0].shape
        self.sequences = [d.reshape(timestep, sensor_dim) for d in data_resampled]
        self.labels = labels_resampled.tolist()

    def print_label_distribution(self):
        # Count the frequency of each label
        label_counts = pd.Series(self.labels).value_counts()
        total_labels = len(self.labels)
        print("Label Distribution:")
        for label, count in label_counts.items():
            print(f"Label {label}: {count} ({(count / total_labels * 100):.2f}%)")


batch_size = 128
# NUM_SENSORS = 14
NUM_SENSORS = 2
NUM_HEAD = 8
# sensors = ['ACC_AXZ','DST_PAC', 'ACC_AXX', 'STP', 'ACC_MAG',
#            'SKT', 'DST_DST','ACC_AXY', 'EDA', 'RRI', 'DST_SPD', 'CAL', 'AML', 'HRT']  # No need to sort them
sensors = ['EDA', 'RRI']  # No need to sort them

participant_labeled_data = list(Path('Intermediate/labeled_joined').glob('*_labeled.csv'))

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

pos_label = 1


average_acc = []
average_f1 = []
average_auc = []
average_precision = []
average_recall = []
average_f1_positive = []
average_f1_negative = []
total_positive_labels = 0
total_negative_labels = 0

for i, (train_index, test_index) in enumerate(kf.split(participant_labeled_data)):
    train_val_files = [participant_labeled_data[idx] for idx in train_index]
    test_files = [participant_labeled_data[idx] for idx in test_index]

    # Now, further split the training data into actual training and validation sets
    # We'll use 80% of train_val_files for training and 20% for validation
    split_idx = int(len(train_val_files) * 0.8)
    train_files = train_val_files[:split_idx]
    val_files = train_val_files[split_idx:]

    train_dataset = MultimodalDataset(train_files, sensors)
    val_dataset = MultimodalDataset(val_files, sensors, include_only_real_labels=True)
    test_dataset = MultimodalDataset(test_files, sensors, include_only_real_labels=True)

    train_dataset.apply_smote()
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = DownstreamModel(embed_dim=batch_size, num_heads=NUM_HEAD, num_sensors=NUM_SENSORS, premodel_states=premodel_states)
    # model = DownstreamModel(embed_dim=batch_size,  num_sensors=NUM_SENSORS, premodel_states=premodel_states) #This is for attention fusion

    model = model.cuda()
    
    tr_ep_loss = []
    tr_ep_acc = []

    val_ep_loss = []
    val_ep_acc = []

    EPOCHS = 10


    dsoptimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(dsoptimizer, step_size=1, gamma=0.98, last_epoch=-1, verbose=True)

    # # Compute class weights for handling class imbalance
    # labels = np.array([label for _, label in train_dataset])
    # class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    # class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()

    # # Initialize the loss function with class weights
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    loss_fn = nn.CrossEntropyLoss()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.001) 


    for epoch in range(EPOCHS):
        stime = time.time()
        print("=============== Epoch : %3d ===============" % (epoch + 1))

        loss_sublist = np.array([])
        acc_sublist = []

        model.train()

        dsoptimizer.zero_grad()

        for batch in train_dataloader:
            x = batch[0]
            y = batch[1]  # y should contain class indices, not one-hot encoded labels
            x = x.cuda().float()
            y = y.cuda().long()  # Convert y to long data type as required

            z = model(x)

            dsoptimizer.zero_grad()

            tr_loss = loss_fn(z, y)  # y is now a tensor of class indices
            tr_loss.backward()

            preds = torch.argmax(z, dim=1).detach().cpu()
            labels = y.cpu()  # y already contains the class indices, so we can use it directly


            dsoptimizer.step()

            loss_sublist = np.append(loss_sublist, tr_loss.cpu().data)
            acc_sublist.extend((preds == labels).tolist())

        print('ESTIMATING TRAINING METRICS.............')

        print('TRAINING BINARY CROSSENTROPY LOSS: ', np.mean(loss_sublist))
        print('TRAINING BINARY ACCURACY: ', np.mean(acc_sublist))

        tr_ep_loss.append(np.mean(loss_sublist))
        tr_ep_acc.append(np.mean(acc_sublist))

        print('ESTIMATING VALIDATION METRICS.............')

        model.eval()

        loss_sublist = np.array([])
        acc_sublist = []

        with torch.no_grad():
            for batch in val_dataloader:
                x = batch[0]
                y = batch[1]
                x = x.cuda().float()
                y = y.cuda().long()  # Ensure that y is a long tensor of class indices

                z = model(x)

                # Calculate loss
                val_loss = loss_fn(z, y)

                # Calculate predictions and probabilities
                preds = torch.argmax(z, dim=1).detach().cpu()
                labels = y.cpu()
                probs = torch.softmax(z, dim=1)[:, 1].detach().cpu()  # Probabilities for the positive class


                loss_sublist = np.append(loss_sublist, val_loss.cpu().data)
                acc_sublist.extend((preds == labels).tolist())

        print('VALIDATION BINARY CROSSENTROPY LOSS: ', np.mean(loss_sublist))
        print('VALIDATION BINARY ACCURACY: ', np.mean(acc_sublist))

        epoch_val_loss = np.mean(loss_sublist)
        print(f"Epoch {epoch + 1} Validation Loss: {epoch_val_loss:.4f}")

        val_ep_loss.append(np.mean(loss_sublist))
        val_ep_acc.append(np.mean(acc_sublist))

        lr_scheduler.step()

        # #The following code is for early stopping
        # early_stopping(epoch_val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        print('Saving model...')
        model_path = 'downstream_models/downstream_unfreeze_pretrained_{}.pt'.format(i)
        torch.save(model.state_dict(), model_path)

        print("Time Taken : %.2f minutes" % ((time.time() - stime) / 60.0))

    # Model Testing

    model = DownstreamModel(embed_dim=batch_size,  num_heads=NUM_HEAD, num_sensors=NUM_SENSORS, premodel_states=premodel_states).cuda() 
    # model = DownstreamModel(embed_dim=batch_size,  num_sensors=NUM_SENSORS, premodel_states=premodel_states).cuda() #This is for attention fusion
    model.load_state_dict(torch.load('downstream_models/downstream_unfreeze_pretrained_{}.pt'.format(i)))

    model.eval()

    test_preds = []
    test_probs = []
    test_labels = []

    fold_positive_labels = 0
    fold_negative_labels = 0

    with torch.no_grad():
        for batch in test_dataloader:
            x = batch[0]
            y = batch[1]
            x = x.cuda().float()
            y = y.cuda().long()  # Ensure that y is a long tensor of class indices

            z = model(x)

            # Calculate loss
            val_loss = loss_fn(z, y)

            # Calculate predictions and probabilities
            preds = torch.argmax(z, dim=1).detach().cpu()
            labels = y.cpu()
            probs = torch.softmax(z, dim=1)[:, 1].detach().cpu()  # Probabilities for the positive class


            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.numpy())
            test_labels.extend(labels.cpu().numpy())
            # Count the number of positive and negative labels in this fold
            fold_positive_labels += torch.sum(labels == pos_label).item()
            fold_negative_labels += torch.sum(labels == 1-pos_label).item()


    fold_f1 = f1_score(test_labels, test_preds, average='macro', pos_label=pos_label)
    fold_accuracy = accuracy_score(test_labels, test_preds)
    fold_auc = roc_auc_score(test_labels, test_probs, average='macro')
    fold_precision = precision_score(test_labels, test_preds, average='macro', pos_label=pos_label)
    fold_recall = recall_score(test_labels, test_preds, average='macro', pos_label=pos_label)
    fold_f1_positive = f1_score(test_labels, test_preds, pos_label=pos_label)
    fold_f1_negative = f1_score(test_labels, test_preds, pos_label=1-pos_label)
    # Accumulate total counts
    total_positive_labels += fold_positive_labels
    total_negative_labels += fold_negative_labels
    
    print(f'TEST FOLD {i}: Accuracy: {fold_accuracy}, F1 Score: {fold_f1}, AUC-ROC: {fold_auc}, Precision: {fold_precision}, Recall: {fold_recall}')
    print(f"Fold {i} Positive Labels: {fold_positive_labels}, Negative Labels: {fold_negative_labels}")


    average_acc.append(fold_accuracy)
    average_f1.append(fold_f1)
    average_auc.append(fold_auc)
    average_precision.append(fold_precision)
    average_recall.append(fold_recall)
    average_f1_positive.append(fold_f1_positive)
    average_f1_negative.append(fold_f1_negative)


    # Print label distribution in test data
    print("Label Distribution in Test Data:")
    print(pd.Series(test_labels).value_counts())

print('TOTAL AVERAGE ACCURACY:', np.mean(average_acc))
print('TOTAL AVERAGE F1:', np.mean(average_f1))
print('TOTAL AVERAGE AUC-ROC:', np.mean(average_auc))
print('TOTAL AVERAGE PRECISION:', np.mean(average_precision))
print('TOTAL AVERAGE RECALL:', np.mean(average_recall))
print('TOTAL AVERAGE F1 (POSITIVE):', np.mean(average_f1_positive))
print('TOTAL AVERAGE F1 (NEGATIVE):', np.mean(average_f1_negative))
# Print total counts across all folds
print(f'Total Positive Labels: {total_positive_labels}, Total Negative Labels: {total_negative_labels}')

