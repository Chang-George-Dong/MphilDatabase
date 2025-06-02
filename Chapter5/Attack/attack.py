import sys

sys.path.append("/home/george/ADMA2024/adma2024/Utils")
sys.path.append("/home/george/ADMA2024/adma2024/Attack")
from models import KAN,MLP,KAN_MLP,MLP_KAN
from attack_methods import *
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


dropout_rate = 0.1
model_names =["KAN","MLP","MLP_L","KAN_MLP","MLP_KAN"]
nets = [KAN,MLP,MLP,KAN_MLP,MLP_KAN]

model_dict = dict(zip(model_names,nets))

Attack_Type = "random"
attacker = pgd_attack_random_target

epses = [0.05,0.1,0.25,0.5]
for eps in epses:
    for dataset in UNIVARIATE_DATASET_NAMES:
        for model_name,net in model_dict.items():
            path = f"{Root_Path}/Attack/All_Model_eps/eps_{eps}/{model_name}/{dataset}"
            try:
                os.makedirs(path)
            except:
                print("Path Already Exists")
                
            result_csv = f'{path}/{Attack_Type}_result.csv'
            if os.path.exists(result_csv):
                print("The path exists.")
                continue
            else:
                print(f"Running: {dataset, model_name, eps}")
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            trainloader, valloader, train_size, val_size, nb_classes = data_loader(dataset, batch_size=4096)
            
            weight_path = f"{Root_Path}/Train_WBN/ALl_Model_Results/{model_name}/{dataset}/model_final.pth"
            size = train_size

            if model_name == "MLP_L":
                model = net([size[-1],size[-1]*10,128, nb_classes], batch_norm =True, dropout_rate=dropout_rate)
            else:
                model = net([size[-1],size[-1],128, nb_classes], batch_norm =True, dropout_rate=dropout_rate)
            checkpoint = torch.load(weight_path)
            model.load_state_dict(checkpoint)
            model.to(device)

            input_size = size[-1]
            model.eval()        

            total_accuracy = 0.0
            total_samples = 0
            total_attack_success = 0
            total_l2_noise = 0.0

            for images, labels in valloader:
                images = images.view(-1, input_size).to(device)
                labels = labels.to(device)
                
                original_output =  model(images)
                _, original_preidicted = original_output.max(1)

                adv_images = attacker(model, images,original_output, eps=eps, alpha=eps*0.01, iters=100)

                outputs = model(adv_images)
                _, predicted = outputs.max(1)
                total_samples += labels.size(0)
                total_accuracy += (predicted == labels).sum().item()

                # Calculate attack success and L2 norm
                total_attack_success += (predicted != original_preidicted).sum().item()
                total_l2_noise += torch.norm((adv_images - images).view(images.size(0), -1), dim=1).sum().item()

            accuracy = total_accuracy / total_samples
            attack_success_rate = total_attack_success / total_samples
            average_l2_noise = total_l2_noise / total_samples

            print(f"Dataset: {dataset}, Model: {model_name},Attack_Type: {Attack_Type},eps: {eps}")
            print(f"Accuracy: {accuracy:.4f}, Attack Success Rate: {attack_success_rate:.4f}, Average L2 Noise: {average_l2_noise:.4f}")
            with open(result_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Dataset", "Model", "Attack_Type","eps", "Accuracy", "Attack Success Rate", "Average L2 Noise"])
                writer.writerow([dataset, model_name,  Attack_Type ,eps, accuracy, attack_success_rate, average_l2_noise])
            print("><><><><><><><><><><><><><><><><><><><><><><><><><")