import queue
import threading

import torch
from sklearn.metrics import multilabel_confusion_matrix
from tqdm import tqdm

from datasets.data_util import to_gpu
from models.lovasz_losses import lovasz_softmax_flat
from scripts.eco_system import GenericEcoSystem
import numpy as np
import matplotlib.pyplot as plt
from scripts.util import evaluate_cm


class TrainingEcoSystem(GenericEcoSystem):
    def __init__(self, config, device, name="Trainer"):
        super().__init__(config=config, device=device, name=name)

    def training_essentials(self):
        assert self.model is not None
        assert self.training_dataset is not None
        assert self.training_dataloader is not None
        assert self.optimizer is not None
        assert self.output_model_path is not None
        assert self.criterion is not None
        assert self.num_classes is not None
        assert self.class_names is not None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def run_training(self):
        self.training_essentials()
        self.epoch = 0
        best_val_miou = 0
        best_val_loss = np.inf
        train_losses = list()
        while self.epoch < self.config['num_epoch']:
            train_loss = self.train_epoch()
            val_miou, val_loss = self.validate_epoch()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
            # save model if performance is improved
            best_val_miou, best_val_loss = self.save_checkpoint(best_val_miou, val_miou, best_val_loss, val_loss)
            self.epoch += 1

        plt.plot(train_losses)
        plt.show()

    def save_checkpoint(self, best_val_miou, val_miou, best_val_loss, val_loss):
        if best_val_miou <= val_miou:
            print(f'Saving model at epoch: {self.epoch}')
            best_val_miou = val_miou
            torch.save(self.model.state_dict(), self.output_model_path)
        if best_val_loss > val_loss:
            print(f'Saving loss model at epoch: {self.epoch}')
            best_val_loss = val_loss
            if self.output_model_loss_path is not None:
                torch.save(self.model.state_dict(), self.output_model_loss_path)
        print('Epoch: %d Current val miou is %.3f  while the best val miou is %.3f' %
              (self.epoch, val_miou, best_val_miou))
        return best_val_miou, best_val_loss

    def move_data_to_gpu(self, batch):
        # ts = time.time_ns()
        data_dict = dict()
        keys = batch.keys()
        Q = queue.Queue()
        threads = list()
        for key in keys:
            param = (batch, key, self.device)
            t = threading.Thread(target=to_gpu, args=(param, Q))
            threads.append(t)

        for t in threads:
            t.start()
            res = Q.get()
            data_dict = data_dict | res

        for t in threads:
            t.join()
        return data_dict

    def handle_forward_pass(self, data_dict, train=True):
        label = data_dict['mapped_label'].long().view(-1)
        if train:
            logit, indices = self.model(data_dict)
        else:
            (logit, _), indices = self.model(data_dict)
        logit = logit[indices]
        label = label[indices]
        return logit, label

    def validate_epoch(self):
        self.model.eval()
        val_loss_list = []
        pcm = np.zeros(shape=(self.num_classes, 2, 2))
        pbar = tqdm(total=len(self.val_dataloader), desc=f'Epoch {self.epoch}')
        with torch.no_grad():
            for i_iter_val, batch in enumerate(
                    self.val_dataloader):
                data_dict = self.move_data_to_gpu(batch)
                logit, label = self.handle_forward_pass(data_dict, train=False)

                loss = self.criterion(logit, label)

                predict_labels = torch.argmax(logit, dim=1)
                mcm = multilabel_confusion_matrix(y_true=label.cpu().numpy(), y_pred=predict_labels.cpu().numpy(),
                                                  labels=[i for i in range(self.num_classes)])
                pcm += mcm
                val_loss_list.append(loss.detach().cpu().numpy())
                pbar.update()

                pbar.set_postfix({'Loss': np.mean(val_loss_list)})
        pbar.close()
        IOUs = list()
        for i, cm in enumerate(pcm):
            if i != 0:
                iou = evaluate_cm(cm, self.class_names[i])
                IOUs.append(iou)
        class_jaccard = torch.tensor(np.array(IOUs))

        val_miou = class_jaccard.sum().item() / class_jaccard.size(0)
        return val_miou, np.mean(val_loss_list)

    def train_epoch(self):
        loss_list = []
        pbar = tqdm(total=len(self.training_dataloader), desc=f'Epoch {self.epoch}')
        # training
        self.model.train()
        pcm = np.zeros(shape=(self.num_classes, 2, 2))
        for i_iter, batch in enumerate(self.training_dataloader):

            data_dict = self.move_data_to_gpu(batch)
            self.optimizer.zero_grad()
            logit, label = self.handle_forward_pass(data_dict)
            try:
                loss = self.criterion(logit, label) + lovasz_softmax_flat(torch.nn.functional.softmax(logit, dim=1),
                                                                          label, ignore=None)
                loss.backward()
                self.optimizer.step()
            except ValueError:
                print(batch['fname'])
            loss_list.append(loss.item())
            pbar.update(1)
            pbar.set_postfix({'Loss': np.mean(loss_list)})
            predict_labels = torch.argmax(logit, dim=-1)
            mcm = multilabel_confusion_matrix(y_true=label.cpu().numpy(), y_pred=predict_labels.cpu().numpy(),
                                              labels=[i for i in range(self.num_classes)])
            pcm += mcm

        pbar.close()
        IOUs = list()
        for i, cm in enumerate(pcm):
            iou = evaluate_cm(cm, self.class_names[i])
            IOUs.append(iou)
        class_jaccard = torch.tensor(np.array(IOUs))
        print('Average iou: ', class_jaccard[1:].mean())
        return np.mean(loss_list)
