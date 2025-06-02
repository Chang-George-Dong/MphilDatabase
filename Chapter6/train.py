import argparse
import copy
import logging
import os
import sys
import time
import yaml
import csv


import numpy as np
import torch
import torch.nn as nn

from model.InceptionTime import InceptionTime
from model.LSTM_FCN import LSTMFCN
from model.MACNN import MACNN
from model.TCN import TCN

from utils import evaluate_standard
from utils import get_loaders


logger = logging.getLogger(__name__)

with open("dataset_configs.yml", "r") as file:
    datasets_info = yaml.safe_load(file)
    dataset_names = list(datasets_info.keys())
def parse_list(value):
    try:
        # Convert the comma-separated string to a list of floats
        return [float(x) for x in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid list value: {value}")
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='./dataset/UCRArchive_2018', type=str)
    parser.add_argument('--dataset', default='Beef',  type=str, choices=dataset_names)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--network', default='InceptionTime', type=str, choices=['InceptionTime', 'LSTMFCN', 'MACNN','TCN'])
    parser.add_argument('--pretrained_path', default=None, type=str, help="Weights transfered from ResNet to ResKAN")
    parser.add_argument('--save_dir', default='ckpt', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--lr_max', default=0.001, type=float)
    parser.add_argument('--lr_min', default=0.0001, type=float)
    parser.add_argument('--device', type=str, default='cuda')  # gpu

    return parser.parse_args()


   
def main():
    args = get_args()

    # 检查 args.dataset 是否在 YAML 数据中
    if args.dataset in datasets_info:
        dataset_info = datasets_info[args.dataset]
        args.num_classes = dataset_info.get("num_classes")
        args.seq_length = int(dataset_info.get("length"))
    else:
        print('Wrong dataset:', args.dataset)
        exit()

    # saving path
    path = os.path.join('ckpt', args.dataset, args.network)
    args.save_dir = os.path.join(path, args.save_dir)
    

    # logger
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if os.path.exists(os.path.join(args.save_dir, 'model_last.pth')):
        print('Task already finished.')
        return 
    else:
        start_epoch = 0
        print('Task Start.>>>>>>')
        logfile = os.path.join(args.save_dir, 'output.log')
        csv_file = os.path.join(args.save_dir, 'output.csv')  

        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc', 'lr'])

        with open(logfile, mode='w',) as f:
            restart_time = time.strftime("%H:%M:%S %d/%m/%Y")
            f.write(f"Task start at {restart_time}\n")
    
    handlers = [logging.FileHandler(logfile, mode='a+'),
                logging.StreamHandler()]
    
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    
    logger.info(args)

    # set current device
    if args.device != "cpu":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    logger.info('Devices: '.format(device.type))

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # get data loader
    train_loader, test_loader = get_loaders(args.dataset, args.batch_size, args.data_dir)

    # setup network
    if args.network == 'InceptionTime':
        net = InceptionTime
    elif args.network == 'LSTMFCN':
        net = LSTMFCN
    elif args.network == 'MACNN':
        net = MACNN
    elif args.network == 'TCN':
        net = TCN
    else:
        print('Wrong network:', args.network)
        exit()

    model = net(num_classes=args.num_classes,  seq_length=args.seq_length).to(device)
    logger.info(model)

    # set weight decay
    params = [{'params': param} for name, param in model.named_parameters()]

    # setup optimizer, loss function, LR scheduler
    opt = torch.optim.Adam(params, lr= args.lr_max)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=50, verbose=True, min_lr=args.lr_min)

    # Start training
    start_train_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        current_lr = opt.param_groups[0]['lr']  # {{ edit_1 }}     
        # logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.train()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            output = model(X)
            loss = criterion(output, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        # Evaluate on test set
        train_loss /= train_n
        train_acc /= train_n
        scheduler.step(train_loss)
        test_loss, test_acc = evaluate_standard(test_loader, model, device)
        
        
        logger.info("Epoch: [{:d}]\ttrain_loss: {:.3f}\ttest_loss: {:.3f}\ttrain_acc: {:.4f}\ttest_acc: {:.4f}\tlr {:.2e}".format(
            epoch,
            train_loss,
            test_loss,
            train_acc,
            test_acc,
            current_lr)
        )
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                test_loss,
                train_acc,
                test_acc,
                current_lr
            ])
        # Save the latest model at the end of each epoch
    torch.save({'state_dict': model.state_dict()}, os.path.join(args.save_dir, 'model_last.pth'))

    elapsed_time = time.time() - start_train_time


    minutes, seconds = divmod(int(elapsed_time), 60)

    logger.info('Training completed in {:02d}m {:02d}s.'.format(minutes, seconds))


if __name__ == '__main__':
    main()
