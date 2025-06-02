from constant import *
import torch
import numpy as np
import os

# DATASET_path = "/Project/2024/pykan/UCR_tasks/UCRArchive_2018"

def readucr(filename, delimiter="\t"):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def map_label(y_data):
    unique_classes, inverse_indices = np.unique(y_data, return_inverse=True)
    mapped_labels = np.arange(len(unique_classes))[inverse_indices]
    return mapped_labels

def load_ucr(dataset, phase="TRAIN"):
    x, y = readucr(os.path.join(DATASET_path, dataset, f"{dataset}_{phase}.tsv"))
    y = map_label(y)
    return x, y

def load_data(dataset, phase="TRAIN", batch_size=128):
    x, y = readucr(os.path.join(DATASET_path, dataset, f"{dataset}_{phase}.tsv"))
    y = map_label(y)
    nb_classes = len(set(y))
    shape = x.shape
    x = x.reshape(shape[0], 1, shape[1])
    x_tensor = torch.tensor(x, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(
        x_tensor, torch.tensor(y, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(phase == "TRAIN")
    )

    return loader, x_tensor.shape, nb_classes


def data_loader(dataset, batch_size=128, train=True, test=True):
    train_loader, train_shape, nb_classes = load_data(
        dataset, "TRAIN", batch_size=batch_size
    )
    test_loader, test_shape, nb_classes = load_data(
        dataset, "TEST", batch_size=batch_size
    )

    return train_loader, test_loader, train_shape, test_shape, nb_classes

def config(structure = None, **kwargs): 
    
    config_dic = {
        "EPOCH": EPOCH,
        "dropout_rate": dropout_rate,
        "weight_decay": weight_decay,
        "lr": lr,
        "factor": factor,
        "patience": patience,
        "reg_A": reg_A,
        "reg_B": reg_B,
        "reg_coeff": reg_coeff,
        "device": str(device),  
        "structure":structure
    }
    return config_dic