import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import os
import uproot as ur
import time
from matplotlib import pyplot as plt

# -------------------------
# Deep Sets model
# -------------------------

class DeepSets(nn.Module):
    """DNN that implements Deep Sets.  We did not find Deep Sets to be more accurate than regular DNNs."""
    
    def __init__(self, ins, outs):
        """
        Args:
            ins(int):, # of input channels
            outs(int): # of output channels"""
        
        super().__init__()

        hidden = 64

        self.phi = nn.Sequential(
            nn.Linear(ins, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU()
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, outs)
        )
        
    def forward(self, x, mask):
        """
        Passes inputs through the net.

        Args:
            x: (B, Nmax, 5)
            mask: (B, Nmax) -> 1 for real particles, 0 for padding

        Returns:
            Net output.
        """

        x = x * mask.unsqueeze(-1)
        
        x = self.phi(x)  # (B, N, 64)
        
        # sum pooling (Deep Sets)
        x = x.sum(dim=1)  # (B, 64)

        x = self.rho(x)

        """
        x = self.layers(x)  # (B, N, 64)
        
        # sum pooling (Deep Sets)
        x = x.sum(dim=1)  # (B, 64)
        """
        return x.view(-1)


class DNN(nn.Module):
    """A regular DNN.  Nothin to see here."""
    
    def __init__(self, ins, outs):
        """Args:
            ins: int, # of input channels
            outs: int, # of output channels"""
        
        super().__init__()

        hidden = 64
        
        self.layers = nn.Sequential(
            nn.Linear(ins, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, outs)
        )

    def forward(self, x, mask):
        """
        Passes inputs through the net.

        Args:
            x: (B, Nmax, 5)
            mask: (B, Nmax) -> 1 for real particles, 0 for padding
        """

        x = x * mask.unsqueeze(-1)

        x = self.layers(x)  # (B, N, 64)

        # sum pooling (Deep Sets)
        x = x.sum(dim=1)  # (B, 64)
        
        return x.view(-1)
# -------------------------
# collate_fn (IMPORTANT)
# -------------------------

def collate_fn(batch):
    """Applies padding and masking to make data batch into an array of regular size.
    
    Returns: padded input, mask (for collate fn), truth, weights, multiplicity, pt"""

    # This commented line was from an older version where we did not pass pt through.
    #xs, ys, ws, ms = zip(*batch)
    xs, ys, ws, ms, pts = zip(*batch)
    
    lengths = torch.tensor([len(x) for x in xs])

    x_pad = pad_sequence(xs, batch_first=True)  # (B, Nmax, 5)

    mask = torch.zeros(x_pad.shape[0], x_pad.shape[1])
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0

    y = torch.stack(ys)
    w = torch.stack(ws)
    m = torch.stack(ms)
    
    #return x_pad, mask, y, w, m
    return x_pad, mask, y, w, m, pts


# -------------------------
# loss
# -------------------------

def qvec_to_cor2(q, mult):
    """Calculates <2> from Q-vector array from NN."""
    
    return (q[::2]**2 + q[1::2]**2 - mult) / (mult * (mult - 1))

def comp_to_mag2(q):
    """Takes the qvector array given by NN and outputs square magnitude."""
    return q[::2]**2 + q[1::2]**2

def loss_fn(pred, truth, mult):
    """Loss is MAE."""
    
    """
    # 2-comp qvec output
    cor2 = qvec_to_cor2(pred, mult)
    return F.l1_loss(cor2, truth.float())
    
    """
    """
    # 2-comp to qmag square
    mag2 = pred[::2]**2 + pred[1::2]**2
    return F.l1_loss(mag2, truth)
    """
    # Compare components
    return F.l1_loss(pred, truth)

    # mae loss
    # return F.l1_loss(pred, truth.float())
    

# -------------------------
# train
# -------------------------

def train(model, loader, optimizer, device):
    """Trains NN.

    Args:
        model: net being trained
        loader: DataLoader
        optimizer: optimizer being used
        device: which device tensors should be stored on.
        
    Returns: loss per batch"""
    
    model.train()
    total = 0
    
    for item in loader:
        x, mask, y, _, mult = item[:5]
        
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device).view(-1)
        mult = mult.to(device)

        optimizer.zero_grad()

        pred = model(x, mask)

        loss = loss_fn(pred, y, mult)

        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / len(loader)


# -------------------------
# validate
# -------------------------

def validate(model, loader, device):
    """Runs validation data through NN.

    Args:
        model: net being trained
        loader: DataLoader
        device: which device tensors should be stored on.
        
    Returns: loss per batch"""
    model.eval()
    total = 0

    with torch.no_grad():
        for item in loader:
            x, mask, y, _, mult = item[:5]

            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device).view(-1)
            mult = mult.to(device)

            pred = model(x, mask).to(device)

            loss = loss_fn(pred, y, mult)

            total += loss.item()

    return total / len(loader)

def split_training_val(dataset):
    train_size = .8
    return dataset[:int(len(dataset) * train_size)], dataset[int(len(dataset) * train_size):]

# -------------------------
# main
# -------------------------

def train_net(num_epochs = 50, batch_size = 128, folders = ("deepset_data",), ins = 3, outs = 2, model_type="dnn", trained_model=None, save=True, name=None, chunks = False):
    """Trains and saves a neural net.
    
    num_epochs(int): num of epochs to train
    batch_size(int): size of training batches in the data loader
    folders: String or list of strings of the names of the datasets.  Datasets should be stored in deepsets/deepset_data/{folder}/data.pt.  If multiple datasets are trained, it will alternate through them, training for one epoch each sequentially.
    ins(int): number of input channels for the net
    outs(int): number of output channels for the net
    model_type(str): This is either \"dnn\" or \"ds\" for DNN or DeepSet.
    trained_model(str): This can be the name of a model that is already trained that will be further trained.
    save(bool): Whether the model should be saved or not.
    name: Can be a string if a specific name for the model is desired.  Otherwise the name will be the same as the datasets given by folders.
    """

    # If a string is inputted into a folder, make it a list just for standardization.
    
    if isinstance(folders, str):
        folders = [folders,]
    folder_id = ""
    for folder in folders:
        folder_id += f"{folder}_"

    # Training-validation split is 80/20.  For the sake of reproducibility, the random split is not activated below.
    
    if not chunks:
        datasets = [torch.load(f"deepsets/deepset_data/{folder}/data.pt") for folder in folders]
        train_datas, val_datas = [], []
        for i in range(len(datasets)):
            #split = random_split(datasets[i], [train_size, val_size])
            split = split_training_val(datasets[i])
            train_datas.append(split[0])
            val_datas.append(split[1])

        train_loaders = [DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) for train_data in train_datas]
        val_loaders = [DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) for val_data in val_datas]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

    # Create model
    if model_type == "ds":
        model = DeepSets(ins, outs).to(device)
    else:
        model = DNN(ins, outs).to(device)
    # Load trained model if desired
    if trained_model is not None:
        model.load_state_dict(torch.load(f"models/{trained_model}.pth", weights_only=True))
    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3)

    start = time.time()

    # Train
    print(folder_id)
    print("Epoch, Training, Validation, Time")
    for epoch in range(num_epochs):
        if not chunks:
            data_index = epoch % len(datasets)
            tr = train(model, train_loaders[data_index], optimizer, device)
            va = validate(model, val_loaders[data_index], device)
        else:
            chunk_i = 0
            while os.path.exists(f"/deepsets/deepset_data/folders[i]_{chunk_i}/data.pth"):
                dataset = torch.load(f"/deepsets/deepset_data/folders[i]_{chunk_i}/data.pth")
                split = split_training_val(dataset)
                tr_loader = DataLoader(split[0], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                val_loader = DataLoader(split[1], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

                tr = train(model, tr_loader, optimizer, device)
                va = validate(model, val_loader, device)
                chunk_i += 1

        print(f"{epoch:03d}, {tr:.5f}, {va:.5f}, {time.time()-start:.1f}s")

    print("Done!")
    
    #Save model according name or dataset
    if save:
        if trained_model is not None:
            folder_id = f"{trained_model}"
        if name is not None:
            folder_id = f"{name}"
        torch.save(model.state_dict(), f"models/{folder_id[:-1]}.pth")
