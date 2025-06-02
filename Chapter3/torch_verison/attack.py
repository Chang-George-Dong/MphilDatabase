import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from classifier import Classifier_INCEPTION
from utils.utils import load_data,write_attack_metrics_to_csv,create_directory
import time
import csv
import json

class Attack(torch.nn.Module):

    def __init__(self, dataset, Model = Classifier_INCEPTION, 
                 batch_size=64,epoch=1000,device=None):
        super(Attack, self).__init__()
        self.model_weight_path = f"result/{dataset}/Done/final_model_weights.pth"
        self.batch_size = batch_size
        self.loader, self.shape, self.nb_classes = load_data(dataset, phase="TEST", batch_size = self.batch_size)
        self.model = Model(input_shape=self.shape, nb_classes=self.nb_classes)
        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_weight_path))
        self.out_dir = f"output/{dataset}/"
        self.dataset = dataset
        self.epoch = epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        os.makedirs(self.out_dir, exist_ok=True)
        
        
    def f(self, x):
        return self.model(x)

    def _loss_function(self, x, y_target):
        y_pred = self.f(x)  
        loss = torch.nn.CrossEntropyLoss()(y_pred, y_target)
        return loss

    def _get_y_target(self, x, method='random'):
        with torch.no_grad():
            y_pred = self.f(x)
            y_target = torch.zeros_like(y_pred)
            if method == 'random':
                _, c = torch.max(y_pred, dim=1)
                for i in range(len(y_pred)):
                    c_s = list(range(y_pred.shape[1]))
                    c_s.remove(c[i].item())
                    new_c = np.random.choice(c_s)
                    y_target[i, new_c] = 1.0
            y_target = torch.argmax(y_target, dim=1)
        return y_target

    def calculate_distance(self,r):
        squared = torch.pow(r, 2)
        sum_distance = torch.sum(squared, dim=2)
        return sum_distance

    def perturb(self, method='random', lr=0.001, eps_init=0.001, eps=0.1):

        if os.path.exists(self.out_dir + "Done"):
            print(f"Task {self.dataset} Already Done")
            print("_______________________________________________________________________________________\n")
            return
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
            print(f"Attack Start: {self.dataset}\n")
            create_directory(self.out_dir + "Doing")

        def to_numpy(tensor):
            return tensor.cpu().detach().numpy()

        total_loss,results_x_adv,results_y_adv,results_y_true= [],[],[],[]
        distances,success_distances, fail_distances = [],[],[]
        
        start_time = time.time()
        for x, y_true in self.loader:
            x, y_true = x.to(self.device), y_true.to(self.device)
            x_adv = x.clone().detach()  # Initialize the adversarial example

            # Initialize the perturbation with random values from {-1, 1} multiplied by eps_init
            r_data = (torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1) * eps_init
            r = r_data.clone().detach().requires_grad_(True)
            y_target = self._get_y_target(x, method=method)

            # Define the optimizer here inside the loop
            optimizer = optim.Adam([r], lr=lr, betas=(0.9, 0.999), eps=1e-07, amsgrad=False)

            batch_loss = []
            for epoch in range(self.epoch):
                x_adv = x + r
                loss = self._loss_function(x_adv, y_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                r.data = torch.clamp(r.data, -eps, eps)
                batch_loss.append(loss.item())

            x_adv = x + r
            y_adv = self.f(x_adv)
            total_loss.append(batch_loss)
                        
            y_pred = y_adv.argmax(-1)
            distance = to_numpy(self.calculate_distance(r))

            success_mask = to_numpy(y_pred != y_true)
            fail_mask = to_numpy(y_pred == y_true)

            distances.append(distance)
            success_distances.append(distance[success_mask])
            fail_distances.append(distance[fail_mask])

            # Collect results for each batch
            results_x_adv.append(x_adv)
            results_y_adv.append(y_adv)
            results_y_true.append(y_true)
            
            del x_adv, y_true, r, batch_loss, loss
            del x, y_adv, y_pred, distance, success_mask, fail_mask
            # Clear GPU cache
            torch.cuda.empty_cache()

        end_time = time.time()

        duration = end_time- start_time
        success_distances = np.concatenate(success_distances)
        fail_distances = np.concatenate(fail_distances)
        all_distances = np.concatenate(distances)

        success_count = success_distances.size
        fail_count = fail_distances.size
        
        mean_success_distance = np.mean(success_distances) if success_distances.size > 0 else None
        mean_fail_distance = np.mean(fail_distances) if fail_distances.size > 0 else None
        mean_all_distance = np.mean(all_distances) if all_distances.size > 0 else None

        ASR = success_distances.size / (success_distances.size + fail_distances.size)
        
        write_attack_metrics_to_csv(self.out_dir + "metrics.csv",ASR,mean_success_distance,mean_fail_distance,
                                    mean_all_distance,success_count,fail_count,duration)

        total_loss = np.array(total_loss) # Convert to a NumPy array

        mean_loss = np.mean(total_loss, axis=0)
        x_ptb = np.concatenate([to_numpy(x)for x in results_x_adv]) 
        x_ptb_reshaped = x_ptb.squeeze(1)
        y_ptb = np.argmax(np.concatenate([to_numpy(y) for y in results_y_adv]),1)

        np.savetxt(self.out_dir+"mean_loss.txt", mean_loss)
        np.savetxt(self.out_dir+"x_perturb.tsv", x_ptb_reshaped, delimiter='\t')
        np.save(self.out_dir+"y_perturb.npy", y_ptb)
        
        os.rename(self.out_dir + "Doing", self.out_dir + "Done")
        print(f"Attack Finished: {self.dataset}\n")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")



                









