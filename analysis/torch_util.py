import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Training and Evaluation Loops
def train_epoch(model, dataloader, criterion, optimizer,device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        if len(targets) == 1:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # model.fc1.apply(clipper)
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            if len(targets) == 1:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_outputs.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.squeeze().cpu().numpy())
    return total_loss / len(dataloader), all_outputs, all_targets


# Convert numpy arrays to PyTorch tensors and create datasets
def create_dataset(x, y):
    x_tensor = torch.Tensor(x).unsqueeze(1)  # Add channel dimension
    y_tensor = torch.Tensor(y).unsqueeze(1)
    y_tensor = y_tensor.type(torch.float32)  # Convert boolean labels to float for BCELoss
    return TensorDataset(x_tensor, y_tensor)
