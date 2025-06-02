import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from classifier import Classifier_INCEPTION
from utils.utils import load_data, write_attack_metrics_to_csv, create_directory
import time
import csv
import json
import pandas as pd


class Attack(torch.nn.Module):
    def __init__(
        self,
        dataset,
        Model=Classifier_INCEPTION,
        batch_size=64,
        epoch=1000,
        eps_init=0.001,
        eps=0.1,
        device=None,
    ):
        super(Attack, self).__init__()
        self.model_weight_path = f"result/{dataset}/Done/final_model_weights.pth"
        self.loader, self.shape, self.nb_classes = load_data(
            dataset, phase="TEST", batch_size=batch_size
        )
        self.model = Model(input_shape=self.shape, nb_classes=self.nb_classes)
        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_weight_path))
        self.out_dir = f"output/{dataset}/"
        self.dataset = dataset
        self.epoch = epoch
        self.eps_init = eps_init
        self.eps = eps
        self.device = device
        self.model.to(self.device)
        os.makedirs(self.out_dir, exist_ok=True)

    def f(self, x):
        return self.model(x)

    def _loss_function(self, x, y_target):
        y_pred = self.f(x)
        loss = torch.nn.CrossEntropyLoss()(y_pred, y_target)
        return loss

    def _get_y_target(self, x, method="random"):
        with torch.no_grad():
            y_pred = self.f(x)
            y_target = torch.zeros_like(y_pred)
            if method == "random":
                _, c = torch.max(y_pred, dim=1)
                for i in range(len(y_pred)):
                    c_s = list(range(y_pred.shape[1]))
                    c_s.remove(c[i].item())
                    new_c = np.random.choice(c_s)
                    y_target[i, new_c] = 1.0
            y_target = torch.argmax(y_target, dim=1)
        return y_target

    def _perturb(self, x, method="random"):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.f(x).detach().cpu().numpy()

        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = r_data.clone().detach().requires_grad_(True)
        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)

        r = torch.nn.Parameter(r_data, requires_grad=True).to(self.device)
        y_target = self._get_y_target(x, method=method).to(
            self.device
        )  # Pass x_device instead of x

        optimizer = optim.Adam(
            [r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False
        )
        sum_losses = np.zeros(self.epoch)
        for epoch in range(self.epoch):
            x_adv = x + r
            loss = self._loss_function(x_adv, y_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)
            sum_losses[epoch] += loss.item()

        x_adv = x + r
        y_adv = self.f(x_adv).argmax(1)

        return x_adv, y_adv, y_pred, sum_losses

    def perturb_all(self):
        start = time.time()
        all_perturbed_x = []
        all_perturbed_y = []
        all_predicted_y = []
        all_sum_losses = np.zeros(self.epoch)

        dist = []
        for x, y in self.loader:
            perturbed_x, perturbed_y, predicted_y, sum_losses = self._perturb(x)
            perturbed_x = perturbed_x.detach().cpu().numpy()
            perturbed_x = np.squeeze(perturbed_x, axis=1)
            dist.append(
                np.sum((perturbed_x - np.squeeze(x.numpy(), axis=1)) ** 2, axis=1)
            )
            all_perturbed_x.append(perturbed_x)

            perturbed_y = perturbed_y.detach().cpu().numpy()
            all_perturbed_y.append(perturbed_y)
            all_predicted_y.append(predicted_y)

            all_sum_losses += sum_losses

        duration = time.time() - start

        x_perturb = np.vstack(all_perturbed_x)
        y_perturb = np.hstack(all_perturbed_y)
        y_predict = np.vstack(all_predicted_y).argmax(axis=1)

        map_ = y_perturb != y_predict
        nb_samples = x_perturb.shape[0]

        Count_Success = sum(map_)
        Count_Fail = nb_samples - Count_Success
        ASR = Count_Success / nb_samples
        distance = np.hstack(dist)
        success_distances = distance[map_]
        failure_distances = distance[~map_]

        # Calculating the means
        mean_success_distance = np.mean(success_distances)
        mean_failure_distance = np.mean(failure_distances)
        overall_mean_distance = np.mean(distance)

        # Create a DataFrame with the data
        data = {
            "ASR": ASR,
            "mean_success_distance": mean_success_distance,
            "mean_failure_distance": mean_failure_distance,
            "overall_mean_distance": overall_mean_distance,
            "Count_Success": Count_Success,
            "Count_Fail": Count_Fail,
            "duration": duration,
        }


        csv_file_path = self.out_dir + "results.csv"
        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data.keys()) 
            writer.writerow(data.values())  

        all_mean_losses = all_sum_losses / nb_samples

        with open(
            os.path.join(self.out_dir, "x_perturb.tsv"), "w", newline=""
        ) as tsv_file:
            writer = csv.writer(tsv_file, delimiter="\t")
            for row in x_perturb:
                writer.writerow(row)

        np.save(os.path.join(self.out_dir, "y_perturb.npy"), y_perturb)

        all_mean_losses = all_mean_losses.reshape(-1, 1)
        np.savetxt(
            os.path.join(self.out_dir, "loss.txt"), all_mean_losses, delimiter="\t"
        )
