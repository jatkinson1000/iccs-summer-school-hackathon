#!/usr/bin/env python3

import numpy as np
import torch
import torch.utils.data as Data
from torch import nn, optim

from L96_model import L96, RK2, RK4, EulerFwd, L96_eq1_xdot, integrate_L96_2t

# Ensuring reproducibility
np.random.seed(14)
torch.manual_seed(14);

# Generate the "real world" model and data with K=8 and J=32
time_steps = 20000
forcing, dt, T = 18, 0.01, 0.01 * time_steps
W = L96(8, 32, F=forcing)
X_true, _, _, xy_true = W.run(dt, T, store=True, return_coupling=True) # xy_true is the effect of Y on X (i.e. U_k)
# Change the data type to `float32` in order to avoid doing type conversions later on
X_true, xy_true = X_true.astype(np.float32), xy_true.astype(np.float32)

# Split the data into training and testing sets
val_size = 4000
X_true_train = X_true[
    :-val_size, :
    ]  # Flatten because we first use single input as a sample
subgrid_tend_train = xy_true[:-val_size, :]
X_true_test = X_true[-val_size:, :]
subgrid_tend_test = xy_true[-val_size:, :]

# Create dataloaders
BATCH_SIZE = 2000 # results in 2 test batches and 8 training batches.
local_dataset = Data.TensorDataset(
    torch.from_numpy(np.reshape(X_true_train, -1)),
    torch.from_numpy(np.reshape(subgrid_tend_train, -1)),
)

local_loader = Data.DataLoader(
    dataset=local_dataset, batch_size=BATCH_SIZE, shuffle=True
)
