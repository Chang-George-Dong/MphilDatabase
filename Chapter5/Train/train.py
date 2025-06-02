import sys

sys.path.append("/home/george/ADMA2024/adma2024/Utils")

from models import KAN,MLP,KAN_MLP,MLP_KAN
from utils import data_loader,config
from constant import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
import time
from sklearn.metrics import f1_score
import csv
import json
import os


lr = 1e-2
dropout_rate = 0.1
EPOCH =1000
model_names =["KAN","MLP","MLP_L","KAN_MLP","MLP_KAN"]
nets = [KAN,MLP,MLP,KAN_MLP,MLP_KAN]
batch_norm = False
model_dict = dict(zip(model_names,nets))




num = 0
for dataset in UNIVARIATE_DATASET_NAMES:
    for model_name,net in model_dict.items():
        
        path = f"{Root_Path}/Train_WOBN/{num}/{model_name}/{dataset}"
        try:
            os.makedirs(path)
        except:
            print("Path Already Exists")

        if os.path.exists(f'{path}/model_final.pth'):
            print("The path exists.")
            continue
        else:
            print(f"Running: {dataset, model_name, num}")
            
        trainloader, valloader, train_size, val_size, nb_classes = data_loader(dataset, batch_size=512)
        # Define model
        size = train_size
        # Initialize the model, optimizer, and scheduler as before
        if model_name == "MLP_L":
            model = net([size[-1],size[-1]*10,128, nb_classes], batch_norm =batch_norm, dropout_rate=dropout_rate)
        else:
            model = net([size[-1],size[-1],128, nb_classes], batch_norm =batch_norm, dropout_rate=dropout_rate)

        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
        
        criterion = nn.CrossEntropyLoss()
        config_dict = config(structure = str(model), **(locals()))

        with open(f"{path}/config.json", "w") as json_file:
            json.dump(config_dict, json_file, indent=4)
        # File to save training logs
        
        log_file = f'{path}/training_log.csv'

        # Initialize log file
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Accuracy', 'Test Accuracy', 'Train F1', 'Test F1', 
                            'Train Loss', 'Test Loss', 'Learning Rate'])

        start_time = time.time()


        for epoch in range(EPOCH):
            # Train
            model.train()
            train_loss = 0
            all_train_labels = []
            all_train_preds = []
            for images, labels in trainloader:
                images = images.view(-1, size[-1]).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                if "MLP" in model_name:
                    total_loss = loss
                else:
                    reg_loss = model.regularization_loss(reg_A, reg_B)
                    total_loss = loss + reg_coeff * reg_loss
                total_loss.backward()
                optimizer.step()

                train_loss += loss.item()
                all_train_labels.extend(labels.cpu().numpy())
                all_train_preds.extend(output.argmax(dim=1).cpu().numpy())
            
            train_loss /= len(trainloader)
            train_accuracy = accuracy_score(all_train_labels, all_train_preds)
            train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')

            # Validation
            model.eval()
            val_loss = 0
            all_val_labels = []
            all_val_preds = []
            with torch.no_grad():
                for images, labels in valloader:
                    images = images.view(-1, size[-1]).to(device)
                    output = model(images)
                    val_loss += criterion(output, labels.to(device)).item()
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(output.argmax(dim=1).cpu().numpy())
            
            val_loss /= len(valloader)
            val_accuracy = accuracy_score(all_val_labels, all_val_preds)
            val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')

            # Update learning rate
            scheduler.step(total_loss)

            # Log epoch results
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, train_accuracy, val_accuracy, train_f1, val_f1, 
                                train_loss, val_loss, optimizer.param_groups[0]['lr']])
            
            # Print epoch results
            # print(dataset,
            #       f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%, "
            #     f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy*100:.2f}%, "
            #     f"Train F1: {train_f1:.4f}, Test F1: {val_f1:.4f}, "
            #     f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save the model only at the last epoch
            if epoch == EPOCH-1:
                torch.save(model.state_dict(), f'{path}/model_final.pth')
            
            end_time = time.time()
            total_runtime = end_time - start_time

        # Save final results to a CSV file
        final_log_file = f'{path}/final_results.csv'
        with open(final_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Train Accuracy', 'Test Accuracy', 'Train F1', 'Test F1', 'Total Runtime'])
            writer.writerow([train_accuracy, val_accuracy, train_f1, val_f1, total_runtime])