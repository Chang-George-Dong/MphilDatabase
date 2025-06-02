import torch
import numpy as np
import os
import torch.nn.functional as F

# DATASET_path = "Dataset/UCRArchive_2018"

def readucr(filename, delimiter="\t"):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

def map_label(y_data):
    unique_classes, inverse_indices = np.unique(y_data, return_inverse=True)
    mapped_labels = np.arange(len(unique_classes))[inverse_indices]
    return mapped_labels

def load_ucr(dataset, DATASET_path = "dataset/UCRArchive_2018", phase="TRAIN"):
    x, y = readucr(os.path.join(DATASET_path, dataset, f"{dataset}_{phase}.tsv"))
    y = map_label(y)
    return x, y

def get_loaders(dataset, batch_size=128, DATASET_path="dataset/UCRArchive_2018",norm=False):
    
    train_x, train_y = load_ucr(dataset, DATASET_path, phase="TRAIN")
    test_x, test_y = load_ucr(dataset, DATASET_path, phase="TEST")
    
    # nb_classes = len(set(train_y))
    
    train_tensor = torch.tensor(train_x, dtype=torch.float32).unsqueeze(1)
    test_tensor = torch.tensor(test_x, dtype=torch.float32).unsqueeze(1)

    if norm:
        train_tensor = train_tensor - train_tensor.mean(dim=1, keepdim=True)
        train_tensor = train_tensor / train_tensor.std(dim=1, keepdim=True)
        test_tensor = test_tensor - test_tensor.mean(dim=1, keepdim=True)
        test_tensor = test_tensor / test_tensor.std(dim=1, keepdim=True)
    
    train_dataset = torch.utils.data.TensorDataset(
        train_tensor, torch.tensor(train_y, dtype=torch.long)
    )
    test_dataset = torch.utils.data.TensorDataset(
        test_tensor, torch.tensor(test_y, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, test_loader


def evaluate_standard(test_loader, model,device):
    """Evaluate without randomization on clean images"""
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n