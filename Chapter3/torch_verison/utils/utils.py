import numpy as np
import torch
import pandas as pd
import os
import csv
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.constant import UNIVARIATE_DATASET_NAMES as datasets


def get_relative_directory(level=0):
    # 获取当前文件所在路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 获取根目录路径
    root_path = current_path
    for _ in range(level):
        root_path = os.path.dirname(root_path)
    return root_path


def readucr(filename, delimiter="\t"):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def map_label(y_data):
    unique_classes, inverse_indices = np.unique(y_data, return_inverse=True)
    mapped_labels = np.arange(len(unique_classes))[inverse_indices]
    return mapped_labels


def load_data(dataset, phase="TRAIN", batch_size=128):
    x, y = readucr(f"./archives/UCRArchive_2018/{dataset}/{dataset}_{phase}.tsv")
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


def metrics(targets, preds):
    precision = precision_score(targets, preds, average="macro", zero_division=0)
    recall = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return precision, recall, f1


def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"'{directory_name}' Created")
    else:
        print(f"'{directory_name}' Existed")


def save_metrics(directory_name, phase, metrics):
    with open(f"{directory_name}/{phase}_metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())


def concat_metrics(mode="train"):
    metrics_dfs = []
    path = os.path.join(get_relative_directory(1), "result", dataset)
    for dataset in datasets:
        file_path = os.path.join(path, f"{mode}_metrics.csv") 
        if os.path.exists(file_path):
            dataset_df = pd.DataFrame([dataset], columns=["dataset"])
            temp_df = pd.read_csv(file_path)
            temp_df = pd.concat([dataset_df] + [temp_df], axis=1)

            metrics_dfs.append(temp_df)

    final_df = pd.concat(metrics_dfs, ignore_index=False)
    final_df.to_csv(f"{mode}_metrics.csv", index=False)


def write_attack_metrics_to_csv(csv_path, *args):
    metrics = {
        "ASR": args[0],
        "Mean Success Distance": args[1],
        "Mean Fail Distance": args[2],
        "Mean All Distance": args[3],
        "Success Count": args[4],
        "Fail Count": args[5],
        "Duration": args[6],
    }

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(metrics.keys())  # Write the metric names (column names)
        writer.writerow(metrics.values())  # Write the corresponding metric values


import numpy as np
import matplotlib.pyplot as plt
from utils.constant import UNIVARIATE_DATASET_NAMES as datasets


def plot_single_line(dataset, mode="learningRate"):
    plt.figure()

    path = os.path.join(get_relative_directory(1), "result", dataset)

    if mode == "learningRate":
        data_path = os.path.join(path, "learningRate.txt")

        data = np.loadtxt(data_path)

    elif mode == "loss":
        data_path = os.path.join(
            get_relative_directory(1), "result", dataset, "learningRate.txt"
        )
        data_path = os.path.join(path, "test_loss.txt")
        data = np.loadtxt(data_path)

    plt.xlabel("epoch")
    plt.ylabel = mode
    plt.plot(data, c="blue")
    plt.title(dataset)
    outdir = os.path.join(path, f"{mode}.png")
    plt.savefig(outdir)


for dataset in datasets:
    plot_single_line(dataset, mode="learningRate")
    plot_single_line(dataset, mode="loss")
