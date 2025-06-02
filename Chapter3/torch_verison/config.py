TEST = False
BATCH_SIZE = 128
EPOCH = 1000
if_Train = True
if_Attack = False


if TEST:
    from utils.constant_test import UNIVARIATE_DATASET_NAMES as DATASETS
else:
    from utils.constant import UNIVARIATE_DATASET_NAMES as DATASETS

import argparse
import torch
from utils.utils import create_directory, concat_metrics
from trainer import Trainer
from attacker import Attack
import tqdm
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    create_directory("result")
    for dataset in tqdm(DATASETS):
        # train_and_evaluate(dataset, device)
        trainer = Trainer(dataset, device=DEVICE, batch_size=BATCH_SIZE, epoch=EPOCH)

        trainer.train_and_evaluate()
    concat_metrics(mode="train")
    concat_metrics(mode="test")


def perform_attack():
    create_directory("output")
    for dataset in tqdm(DATASETS):
        model = Attack(
            dataset=dataset, device=DEVICE, batch_size=BATCH_SIZE, epoch=EPOCH
        )
        model.perturb_all()
