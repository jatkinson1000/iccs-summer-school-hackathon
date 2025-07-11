#!/usr/bin/env python3

"""
L96_training.py

This script trains a neural network to emulate the subgrid tendencies in the Lorenz-96
(L96) model. It generates data using the two-timescale L96 model, and uses this to train
a fully connected neural network to predict the subgrid tendencies. The trained model is
saved as a TorchScript file for later use.
"""

import time

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
l96_model = L96(8, 32, F=forcing)
X_true, _, _, xy_true = l96_model.run(dt, T, store=True, return_coupling=True) # xy_true is the effect of Y on X (i.e. U_k)
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
# Training
dataset_train = Data.TensorDataset(
    torch.from_numpy(X_true_train),
    torch.from_numpy(subgrid_tend_train),
)
loader_train = Data.DataLoader(
    dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True
)
# Test
dataset_test = Data.TensorDataset(
    torch.from_numpy(X_true_test),
    torch.from_numpy(subgrid_tend_test),
)

loader_test = Data.DataLoader(
    dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True
)

# MSE loss function
loss_fn = torch.nn.MSELoss()

# Define the network
class NonLocal_FCNN(nn.Module):
    """
    Fully Connected Neural Network for emulating nonlocal subgrid tendencies in the L96 model.

    Architecture:
        - Input layer: 8 features (resolved variables)
        - Hidden layers: 2 layers with 16 neurons each, ReLU activation
        - Output layer: 8 features (subgrid tendencies)
    """
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 16)  # 8 inputs
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, 8)  # 8 outputs

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 8)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 8)
        """
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

nonlocal_fcnn_network = NonLocal_FCNN()

# Optimiser
learning_rate = 0.003
optimizer_nonlocal_fcnn = optim.Adam(
    nonlocal_fcnn_network.parameters(), lr=learning_rate
)

def train_model(network, criterion, loader, optimizer):
    """
    Train the network for one epoch
    
    Args:
        network (nn.Module): The neural network to train.
        criterion (nn.Module): Loss function.
        loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating network weights.

    Returns:
        float: Average training loss for the epoch.
    """
    network.train()

    train_loss = 0
    for batch_x, batch_y in loader:
        # Get predictions
        if len(batch_x.shape) == 1:
            # This if block is needed to add a dummy dimension if our inputs are 1D
            # (where each number is a different sample)
            prediction = torch.squeeze(network(torch.unsqueeze(batch_x, 1)))
        else:
            prediction = network(batch_x)

        # Compute the loss
        loss = criterion(prediction, batch_y)
        train_loss += loss.item()

        # Clear the gradients
        optimizer.zero_grad()

        # Backpropagation to compute the gradients and update the weights
        loss.backward()
        optimizer.step()

    return train_loss / len(loader)

def test_model(network, criterion, loader):
    """
    Evaluate the neural network on the test dataset.

    Args:
        network (nn.Module): The neural network to evaluate.
        criterion (nn.Module): Loss function.
        loader (DataLoader): DataLoader for test data.

    Returns:
        float: Average test loss.
    """
    network.eval()  # Evaluation mode (important when having dropout layers)

    test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            # Get predictions
            if len(batch_x.shape) == 1:
                # This if block is needed to add a dummy dimension if our inputs are 1D
                # (where each number is a different sample)
                prediction = torch.squeeze(network(torch.unsqueeze(batch_x, 1)))
            else:
                prediction = network(batch_x)

            # Compute the loss
            loss = criterion(prediction, batch_y)
            test_loss += loss.item()

        # Get an average loss for the entire dataset
        test_loss /= len(loader)

    return test_loss

def fit_model(network, criterion, optimizer, train_loader, test_loader, n_epochs):
    """
    Train and validate the neural network for a specified number of epochs.

    Args:
        network (nn.Module): The neural network to train.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating network weights.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        n_epochs (int): Number of training epochs.

    Returns:
        tuple: (list of training losses, list of test losses)
    """
    train_losses, test_losses = [], []
    start_time = time.time()
    for epoch in range(1, n_epochs + 1):
        train_loss = train_model(network, criterion, train_loader, optimizer)
        test_loss = test_model(network, criterion, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    end_time = time.time()
    print(f"Training completed in {int(end_time - start_time)} seconds.")

    return train_losses, test_losses

# Train the model
n_epochs = 120
train_loss, test_loss = fit_model(
    nonlocal_fcnn_network,
    loss_fn,
    optimizer_nonlocal_fcnn,
    loader_train,
    loader_test,
    n_epochs,
)

# Save network
scripted_model = torch.jit.script(nonlocal_fcnn_network)
scripted_model.save("L96_emulator.pt")
