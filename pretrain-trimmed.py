from pathlib import Path
import argparse


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

from torch.utils.data import Dataset, DataLoader

class UnimodalDataset(Dataset):
    def __init__(self, path:Path, sensor_name:str):
        self.path = path
        self.sensor_name = sensor_name
        df = pd.read_csv(self.path, index_col='timestamp')
        self.sequences = df.loc[:,sensor_name]
    
    def augment_sequence(self, original_sequence: pd.Series) -> np.ndarray:
        # Perform agumentations
        
        def jitter(data, jitter_range):
            return np.roll(data, np.random.randint(-jitter_range, jitter_range))
        
        def scale(data, scale_factor=0.8):
          return data * scale_factor

        jittered = jitter(original_sequence, 6)
        scaled = scale(jittered, np.random.uniform(low=.5, high=0.95))

        return scaled



    def __getitem__(self, index) -> tuple[torch.Tensor,torch.Tensor]:
        sequence_index = self.sequences.index.unique()
        original_sequence = self.sequences.loc[sequence_index[index]]
        augmented_sequence = self.augment_sequence(original_sequence)
        return torch.Tensor(original_sequence).view(1, -1), torch.Tensor(augmented_sequence).view(1, -1)


    def __len__(self):
        return len(self.sequences.index.unique())
    


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

class ProjectionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectionHead, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Softmax()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)

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

class PreModel(nn.Module):
    def __init__(self, latent_size):
        super(PreModel, self).__init__()

        self.encoder = Encoder()
        self.projection = ProjectionHead(128, latent_size)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)

        return x
    

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss
    
from torch.optim.optimizer import Optimizer, required
import re

EETA_DEFAULT = 0.001


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=0.0,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=EETA_DEFAULT,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.gt(0),
                        torch.where(
                            g_norm.gt(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True
    
model = PreModel(latent_size=128).cuda()

batch_size = 128

optimizer = LARS(
   [params for params in model.parameters() if params.requires_grad],
   lr=0.2,
   weight_decay=1e-6,
   exclude_from_weight_decay=["bias"],
)

#warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)

#SCHEDULER FOR COSINE DECAY
mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)

#LOSS FUNCTION
criterion = SimCLR_Loss(batch_size = batch_size, temperature = 0.5)


def save_model(model, sensor_name):
    out = Path(f'pretrained_models/{sensor_name}.pt')

    torch.save(model.state_dict(), out)

def train_participant(participant_dataloader: DataLoader, sensor_name: str):

    
    epochs = 10
    tr_loss = []

    for epoch in range(epochs):
            
        print(f"Epoch [{epoch}/{epochs}]\t")
        stime = time.time()

        model.train()
        tr_loss_epoch = 0
        
        for step, (x_i, x_j) in enumerate(participant_dataloader):
            if x_i.size(0) != batch_size:
                break
            optimizer.zero_grad()
            x_i = x_i.cuda()
            x_j = x_j.cuda()

            # positive pair, with encoding
            z_i = model(x_i)
            z_j = model(x_j)

            loss = criterion(z_i, z_j)
            loss.backward()

            optimizer.step()
            
            loss_item = loss.item()
            if step % 2 == 0:
                print(f"Step [{step}/{len(participant_dataloader)}]\t Loss: {round(loss_item, 5)}")

            tr_loss_epoch += loss_item

        
        mainscheduler.step()

        
        time_taken = (time.time()-stime)/60
        print(f"Epoch [{epoch}/{epochs}]\t Time Taken: {time_taken} minutes \t Loss: {tr_loss_epoch}")

    save_model(model, sensor_name)



parser = argparse.ArgumentParser()
parser.add_argument("--sensor_name", type=str, required=True)
args = parser.parse_args()

sensor_name = args.sensor_name
for participant in Path('Intermediate/proc/unlabeled_joined').iterdir():
    participant_dataset = UnimodalDataset(participant, sensor_name)
    participant_dataloader = DataLoader(participant_dataset, batch_size=batch_size, shuffle=False)
    train_participant(participant_dataloader, sensor_name)



