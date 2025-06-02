import os
import torch
import time
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from classifier import Classifier_INCEPTION
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import data_loader, metrics,  concat_metrics, create_directory,save_metrics
from datetime import datetime

class Trainer:
    def __init__(self,  dataset, device, batch_size, epoch = 750, loss = CrossEntropyLoss):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.directory_name = f"./train_result/{dataset}"
        create_directory(self.directory_name)
        self.old_directory_name = f"{self.directory_name}/Doing"
        self.new_directory_name = f"{self.directory_name}/Done"
        self.epoch = epoch
        train_loader, test_loader, shape,_, nb_classes = data_loader(dataset,batch_size=self.batch_size)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = Classifier_INCEPTION(input_shape=shape, nb_classes=nb_classes)
        self.model.to(device)

        self.loss_function = loss()
        self.optimizer = Adam(self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=50, verbose=True, min_lr=0.0001)

    def train_and_evaluate(self):
        if os.path.exists(self.new_directory_name):
            print(f"Task {self.dataset} Already Done")
            print("_______________________________________________________________________________________\n")
            return
        else:
            print(f"Calculating Task {self.dataset}")
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
            create_directory(self.old_directory_name)

        start_epoch, latest_checkpoint_path = self._load_weights()
        if latest_checkpoint_path:
            checkpoint = torch.load(latest_checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Resuming from epoch {start_epoch}")

        test_loss_file = open(f"{self.directory_name}/test_loss.txt", "a")
        learning_rate_file = open(f"{self.directory_name}/learningRate.txt", "a")

        duration = 0
        current_time = datetime.now()
        print("Current time:", current_time,"\n")

        for epoch in range(start_epoch, self.epoch):
            self.model.train()
            start_time = time.time()
            train_metrics = self.train_one_epoch(self.model, self.train_loader, self.device, self.loss_function, self.optimizer)
            duration += time.time() - start_time
            # Evaluation Phase
            train_metrics["duration"] = duration
            test_metrics = self.evaluate(self.model, self.test_loader, self.device, self.loss_function)

            # Record test loss and learning rate
            test_loss_file.write(f"{test_metrics['loss']}\n")
            learning_rate_file.write(f"{self.optimizer.param_groups[0]['lr']}\n")
            self.scheduler.step(test_metrics['loss'])

            # Save model weights every 50 epochs and delete the old one
            if (epoch+1) % 50 == 0:
                checkpoint_path = f"{self.old_directory_name}/epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                }, checkpoint_path)
                old_checkpoint_path = f"{self.old_directory_name}/epoch_{epoch - 50}.pth"
                if os.path.exists(old_checkpoint_path):
                    os.remove(old_checkpoint_path)

        # Final steps after training
        final_weight_path = f"{self.old_directory_name}/final_model_weights.pth"
        torch.save(self.model.state_dict(), final_weight_path)


        for filename in os.listdir(self.old_directory_name):
            if filename.endswith('.pth') and filename != 'final_model_weights.pth':
                file_path = os.path.join(self.old_directory_name, filename)
                os.remove(file_path)


        save_metrics(self.directory_name, 'train', train_metrics)
        save_metrics(self.directory_name, 'test', test_metrics)
        test_loss_file.close()
        learning_rate_file.close()

        # Rename the directory
        os.rename(self.old_directory_name, self.new_directory_name)
        print(f"Task {self.dataset} Finished")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

    def _load_weights(self):
        start_epoch = 0
        latest_checkpoint_path = None
        for epoch in range(0, self.epoch, 50):
            checkpoint_path = f"{self.old_directory_name}/epoch_{epoch-1}.pth"
            if os.path.exists(checkpoint_path):
                latest_checkpoint_path = checkpoint_path
                start_epoch = epoch
        return start_epoch, latest_checkpoint_path

    def evaluate(self, model, test_loader, device, loss_function):
        model.eval()
        test_loss = 0
        correct = 0
        test_preds, test_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions = model(x_batch)
                test_loss += loss_function(predictions, y_batch).item()
                pred = predictions.argmax(dim=1, keepdim=True)
                correct += pred.eq(y_batch.view_as(pred)).sum().item()
                test_preds.extend(pred.squeeze().cpu().numpy())
                test_targets.extend(y_batch.cpu().numpy())
            
            test_loss /= len(test_loader)
            
        accuracy = correct / len(test_loader.dataset)
        precision,recall,f1 = metrics(test_targets, test_preds)

        return {'loss': test_loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def train_one_epoch(self, model, train_loader, device, loss_function, optimizer):
        train_loss = 0
        train_preds, train_targets = [], []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(predictions.argmax(dim=1).cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())
            
        train_loss /= len(train_loader)
        
        accuracy = np.mean(np.array(train_preds) == np.array(train_targets))
        precision,recall,f1 =  metrics(train_targets, train_preds)
        
        return {'loss': train_loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


