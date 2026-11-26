import os

import torch
import numpy as np
from datasets.collate import collate_fn_generic


class GenericEcoSystem(object):
    def __init__(self, config, device, name="Generic Eco System"):
        self.config = config
        self.num_classes = None
        self.class_names = None
        self.process_config()
        self.device = device
        self.name = name
        self.model = None
        self.training_dataset = None
        self.training_dataloader = None
        self.val_dataset = None
        self.val_dataloader = None
        self.test_dataset = None
        self.test_dataloader = None
        self.epoch = 0
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.output_model_path = None
        self.output_model_loss_path = None



    def process_config(self):
        class_strings = self.config["labels"]
        class_inv_remap = self.config["learning_map_inv"]
        self.num_classes = len(class_inv_remap)

        keys = class_inv_remap.keys()
        max_key = max(keys)
        look_up_table = np.zeros((max_key + 1), dtype=np.int32)
        for k, v in class_inv_remap.items():
            look_up_table[k] = v

        self.class_names = [class_strings[class_inv_remap[i]] for i in range(self.num_classes)]

    def set_model(self, model):
        self.model = model

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler

    def set_output_model_path(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.output_model_path = path

    def set_output_model_loss_path(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.output_model_loss_path = path
    def set_training_dataset(self, dataset):
        self.training_dataset = dataset
        self.training_dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                               batch_size=self.config['train_batch_size'],
                                                               shuffle=True,
                                                               num_workers=12,
                                                               collate_fn=collate_fn_generic)

    def set_validation_dataset(self, dataset):
        self.val_dataset = dataset
        self.val_dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                          batch_size=self.config['val_batch_size'],
                                                          shuffle=False,
                                                          num_workers=12,
                                                          collate_fn=collate_fn_generic)

    def set_test_dataset(self, dataset):
        self.test_dataset = dataset
        self.test_dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                           batch_size=self.config['val_batch_size'],
                                                           shuffle=False,
                                                           num_workers=12,
                                                           collate_fn=collate_fn_generic)

    def run_assertion(self):
        assert self.model is not None
        assert self.training_dataset is not None
        assert self.training_dataloader is not None

    def run_training(self):
        pass
