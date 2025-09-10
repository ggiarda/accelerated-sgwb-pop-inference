import random
import json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, random_split


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0) 

def make_input_format_torch(lambdas, freqs=None):

    if freqs is None:
        freqs = np.linspace(10, 2000, num=100)
    elif np.isscalar(freqs):
        freqs = np.array([freqs])
    elif not isinstance(freqs, np.ndarray):
        freqs = np.array(freqs)
    
    log_freqs = np.log(freqs)
    num_freqs = len(freqs)
    
    lambdas = np.array(lambdas)
    
    if lambdas.ndim == 1:
        lambdas = lambdas.reshape(1, -1)
        
    num_lambdas = lambdas.shape[0]
    
    input_array = np.empty((num_freqs * num_lambdas, lambdas.shape[1] + 1))
    
    input_array[:, 1:] = np.repeat(lambdas, num_freqs, axis=0)
    input_array[:, 0] = np.tile(log_freqs, num_lambdas)
    
    return input_array

freqs    = np.logspace(np.log(10), np.log(2000), num=400, base=np.e)

with open('../../data/plpp/train.json','r') as f:
    data = json.load(f)
    
labels = data['labels']
Xtrain = np.array(data['X_train'])
ytrain = np.array(data['y_train'])

ytrain = np.log(ytrain.transpose(0, 2, 1).reshape(-1, 2))
Xtrain = make_input_format_torch(Xtrain, freqs)

ytrain = torch.tensor(ytrain, dtype=torch.float32)[:, 0].unsqueeze(1)
Xtrain = torch.tensor(Xtrain, dtype=torch.float32)


input_size = len(Xtrain[0])
output_size = len(ytrain[0])

output_mean = torch.mean(ytrain, dim=0)
output_std = torch.std(ytrain, dim=0)
ytrain_normalized = (ytrain - output_mean)/output_std

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_size)

        self.input_mean = None
        self.input_std = None

        self.output_mean = None
        self.output_std = None

    def normalize_input(self, x):
        if self.input_mean is not None and self.input_std is not None:
            x = (x - self.input_mean) / self.input_std
        return x
    
    def denormalize_output(self, x):
        if self.output_mean is not None and self.output_std is not None:
            x = (x * self.output_std) + self.output_mean
        return x

    def forward(self, x):

        x = self.normalize_input(x)

        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        
        x = self.output(x)

        x = self.denormalize_output(x)

        return x
    
    def compute_input_normalization_params(self, x):
        self.input_mean = torch.mean(x, dim=0)
        self.input_std = torch.std(x, dim=0)

    def compute_output_normalization_params(self, x):
        self.output_mean = torch.mean(x, dim=0)
        self.output_std = torch.std(x, dim=0)

class EarlyStopper:
    def __init__(self, patience=3, tol=0.0, model=None, checkpoint_path="best_model.pth"):
        self.patience = patience
        self.tol = tol
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.model = model  # Store reference to the model
        self.checkpoint_path = checkpoint_path  # Filepath for saving the best model

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss * (1 - self.tol):
            self.min_validation_loss = validation_loss
            self.counter = 0
            if self.model is not None:
                torch.save(self.model, self.checkpoint_path)  # Save best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Restoring best model before stopping...")
                if self.model is not None:
                    self.model = torch.load(self.checkpoint_path)  # Restore best model
                return True
        return False
    
model = MLP(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

validation_split = 0.2
batch_size = 128
epochs = 1000

dataset = TensorDataset(Xtrain, ytrain_normalized)

val_size = int(validation_split * len(dataset))
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model.compute_input_normalization_params(Xtrain)

early_stopper = EarlyStopper(patience=20, tol=1e-4, model=model)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


train_losses = []
val_losses = []

# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        train_loss += loss.item() * inputs.size(0)  # Accumulate training loss

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)  # Accumulate validation loss

    # Average losses over the entire dataset
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)

    # Save the losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Print losses for the current epoch
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Step the scheduler
    scheduler.step(val_loss)

    if early_stopper.early_stop(val_loss):
        print(f'Early stopping at epoch {epoch + 1}')
        break

#torch.save(model, 'mlp_plpp.pth')
print("Loading the best saved model...")
model = torch.load(early_stopper.checkpoint_path)

model.compute_output_normalization_params(ytrain)

torch.save(model, 'mlp_plpp.pth')

# Save the training data to a JSON file after training
training_data = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "validation_split": validation_split,
    "batch_size": batch_size,
    "epochs": epochs
}

with open('training_data.json', 'w') as f:
    json.dump(training_data, f, indent=4)