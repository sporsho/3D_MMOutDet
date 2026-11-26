#!/usr/bin/env python3
import os
import argparse
import queue
import sys
import threading

import numpy as np
import torch
from qhoptim.pyt import QHAdam
import yaml
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix
import warnings
import matplotlib.pyplot as plt
from datasets.collate import collate_fn_generic
from datasets.data_util import to_gpu
from datasets.snowy_scenes import FusedDataSetSimplified
from models import MidFusedOutDet_PCD_Only
from models.lovasz_losses import lovasz_softmax_flat
from scripts.deterministic import configure_randomness
from scripts.eco_system import TrainingEcoSystem
from scripts.util import evaluate_cm, LovaszBinarySeg, GenericCriterion

warnings.filterwarnings("ignore")


class M05Trainer(TrainingEcoSystem):
    def __init__(self, config, device, name="M05 Trainer"):
        super().__init__(config, device, name)


def main(args):
    # process the arguments
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    model_save_path = args.model_save_path
    output_model_path = os.path.join(model_save_path, 'outdet.pt')
    output_loss_model_path = os.path.join(model_save_path, 'outdet_loss.pt')
    device = torch.device(args.device)

    # load the config file
    with open(args.label_config, 'r') as stream:
        config = yaml.safe_load(stream)
    # move some args to config
    config['val_batch_size'] = val_batch_size
    config['train_batch_size'] = train_batch_size
    config['num_epoch'] = args.num_epoch

    # create the trainer
    trainer = M05Trainer(config=config, device=device, name="M05 Trainer")
    trainer.set_output_model_path(output_model_path)
    trainer.set_output_model_loss_path(output_loss_model_path)

    # prepare model
    model = MidFusedOutDet_PCD_Only(num_classes=trainer.num_classes, kernel_size=args.K, depth=1, dilate=1, width=256)
    model = model.to(device)
    # send the model to trainer
    trainer.set_model(model)

    # create optimizer
    optimizer = QHAdam(trainer.model.parameters(), lr=1e-3, nus=(0.7, 1.0), betas=(0.9, 0.999))
    # send the optimizer to trainer
    trainer.set_optimizer(optimizer)
    # create scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=trainer.optimizer, milestones=[30, 40, 45], gamma=0.1)
    # send the scheduler ot trainer
    trainer.set_scheduler(scheduler)

    # create criterion of CE and Lovasz softmax loss
    primary_criterion = torch.nn.CrossEntropyLoss()
    secondary_criterion = LovaszBinarySeg()
    criterion = GenericCriterion(primary=primary_criterion, secondary=secondary_criterion)
    # send the criterion to trainer
    trainer.set_criterion(criterion)

    # prepare dataset
    tree_k = int(np.round(args.K * args.K))

    train_dataset = FusedDataSetSimplified(data_root=args.data_dir, data_set='train', device=device, k=tree_k,
                                           recalculate=False, falling_only=False,
                                           label_conf='../../datasets/binary_desnow_rs.yaml',
                                           limit=100, drop_thermal=False, drop_rgb=False, )

    val_dataset = FusedDataSetSimplified(data_root=args.data_dir, data_set='val', device=device, k=tree_k,
                                         recalculate=False, falling_only=False,
                                         label_conf='../../datasets/binary_desnow_rs.yaml',
                                         limit=100, drop_thermal=False, drop_rgb=False, )
    # send the data to trainer
    trainer.set_training_dataset(train_dataset)
    trainer.set_validation_dataset(val_dataset)


    # training
    trainer.run_training()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default="/var/local/home/aburai/DATA/ROADVIEW5k")
    parser.add_argument("--label_config", type=str, default='../../datasets/binary_desnow_rs.yaml')
    parser.add_argument('-p', '--model_save_path',
                        default='/var/local/home/aburai/DATA/Fusion_Exp/MID/RS5K/3D_OutDet_P_run2')
    parser.add_argument('--K', type=int, default=3, help='batch size for training (default: 2)')

    parser.add_argument('--train_batch_size', type=int, default=1, help='batch size for training (default: 2)')
    parser.add_argument('--val_batch_size', type=int, default=1, help='batch size for validation (default: 2)')
    parser.add_argument('--device', type=str, default='cuda:1', help='validation interval (default: 4000)')
    parser.add_argument('--num_epoch', type=int, default=50, help='')

    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    configure_randomness(12345)
    main(args)
