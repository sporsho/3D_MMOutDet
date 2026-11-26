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
from scripts.util import evaluate_cm

warnings.filterwarnings("ignore")

def main(args):
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    model_save_path = args.model_save_path
    output_model_path = os.path.join(model_save_path, 'outdet.pt')
    output_loss_model_path = os.path.join(model_save_path, 'outdet_loss.pt')
    os.makedirs(model_save_path, exist_ok=True)
    device = torch.device(args.device)
    dilate = 1

    # prepare miou fun
    with open(args.label_config, 'r') as stream:
        config = yaml.safe_load(stream)

    class_strings = config["labels"]
    class_inv_remap = config["learning_map_inv"]
    # value for masking labels which are not labeled.
    num_classes = len(class_inv_remap)
    ordered_class_names = [class_strings[class_inv_remap[i]] for i in range(num_classes)]

    # channels_to_use = len(rgb_ind)

    # prepare model
    model = MidFusedOutDet_PCD_Only(num_classes=num_classes, kernel_size=args.K, depth=1, dilate=dilate, width=256 )
    model = model.to(device)

    optimizer = QHAdam(model.parameters(), lr=1e-3, nus=(0.7, 1.0), betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 40, 45], gamma=0.1)

    criterion = torch.nn.CrossEntropyLoss()
    # prepare dataset
    tree_k = int(np.round(args.K * args.K))
    train_dataset = FusedDataSetSimplified(data_root=args.data_dir, data_set='train', device=device, k=tree_k,
                                           recalculate=False, falling_only=False,
                                           label_conf='../../datasets/binary_desnow_rs.yaml',
                                           limit=None, drop_thermal=False, drop_rgb=False,)

    val_dataset = FusedDataSetSimplified(data_root=args.data_dir, data_set='val', device=device, k=tree_k,
                                         recalculate=False, falling_only=False,
                                         label_conf='../../datasets/binary_desnow_rs.yaml',
                                         limit=None, drop_thermal=False, drop_rgb=False, )

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_batch_size,
                                                       shuffle=True,
                                                       num_workers=12,
                                                       collate_fn=collate_fn_generic)
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_batch_size,
                                                     shuffle=False,
                                                     num_workers=12,
                                                     collate_fn=collate_fn_generic)

    # training

    # training
    epoch = 0
    best_val_miou = 0
    best_val_loss = np.inf
    train_losses = list()
    while epoch < args.num_epoch:
        train_loss = train_epoch(epoch, model, train_dataset_loader, criterion, optimizer, num_classes,
                                 ordered_class_names, device)
        val_miou, val_loss = validate_epoch(epoch, model, val_dataset_loader, criterion, num_classes, ordered_class_names, device)
        train_losses.append(train_loss)
        scheduler.step()
        # save model if performance is improved
        if best_val_miou <= val_miou:
            print(f'Saving model at epoch: {epoch}')
            best_val_miou = val_miou
            torch.save(model.state_dict(), output_model_path)
        if best_val_loss > val_loss:
            print(f'Saving loss model at epoch: {epoch}')
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_loss_model_path)
        print('Epoch: %d Current val miou is %.3f  while the best val miou is %.3f' %
              (epoch, val_miou, best_val_miou))
        epoch += 1

    plt.plot(train_losses)
    plt.show()


def train_epoch(epoch, model, train_dataset_loader, criterion, optimizer, n_classes, class_names, device):
    loss_list = []
    pbar = tqdm(total=len(train_dataset_loader), desc=f'Epoch {epoch}')
    # training
    model.train()
    pcm = np.zeros(shape=(n_classes, 2, 2))
    for i_iter, batch in enumerate(train_dataset_loader):
        data_dict = dict()
        # ts = time.time_ns()
        keys = batch.keys()
        Q = queue.Queue()
        threads = list()
        for key in keys:
            param = (batch, key, device)
            t = threading.Thread(target=to_gpu, args=(param, Q))
            threads.append(t)

        for t in threads:
            t.start()
            res = Q.get()
            data_dict = data_dict | res

        for t in threads:
            t.join()
        label = data_dict['mapped_label'].long().view(-1)

        optimizer.zero_grad()
        logit, indices = model(data_dict)
        logit = logit[indices]
        label = label[indices]
        try:
            loss = criterion(logit, label) + lovasz_softmax_flat(torch.nn.functional.softmax(logit, dim=1),
                                                                         label, ignore=None)
            loss.backward()
            optimizer.step()
        except ValueError:
            print(batch['fname'])

        loss_list.append(loss.item())
        # zero the parameter gradients

        pbar.update(1)
        pbar.set_postfix({'Loss': np.mean(loss_list)})
        predict_labels = torch.argmax(logit, dim=-1)
        mcm = multilabel_confusion_matrix(y_true=label.cpu().numpy(), y_pred=predict_labels.cpu().numpy(),
                                          labels=[i for i in range(n_classes)])
        pcm += mcm

    pbar.close()
    IOUs = list()
    for i, cm in enumerate(pcm):
        iou = evaluate_cm(cm, class_names[i])
        IOUs.append(iou)
    class_jaccard = torch.tensor(np.array(IOUs))
    print('Average iou: ', class_jaccard[1:].mean())
    return np.mean(loss_list)


def validate_epoch(epoch, model, val_dataset_loader, criterion, n_classes, class_names, device):
    model.eval()
    val_loss_list = []
    pcm = np.zeros(shape=(n_classes, 2, 2))
    pbar = tqdm(total=len(val_dataset_loader), desc=f'Epoch {epoch}')
    with torch.no_grad():
        for i_iter_val, batch in enumerate(
                val_dataset_loader):
            data_dict = dict()
            # ts = time.time_ns()
            keys = batch.keys()
            Q = queue.Queue()
            threads = list()
            for key in keys:
                param = (batch, key, device)
                t = threading.Thread(target=to_gpu, args=(param, Q))
                threads.append(t)

            for t in threads:
                t.start()
                res = Q.get()
                data_dict = data_dict | res

            for t in threads:
                t.join()
            label = data_dict['mapped_label'].long().view(-1)
            (logit, _), indices = model(data_dict)
            logit = logit[indices]
            label = label[indices]

            loss = criterion(logit, label) + lovasz_softmax_flat(torch.nn.functional.softmax(logit, dim=1),
                                                             label, ignore=None)

            predict_labels = torch.argmax(logit, dim=1)
            mcm = multilabel_confusion_matrix(y_true=label.cpu().numpy(), y_pred=predict_labels.cpu().numpy(),
                                              labels=[i for i in range(n_classes)])
            pcm += mcm
            val_loss_list.append(loss.detach().cpu().numpy())
            pbar.update()

            pbar.set_postfix({'Loss': np.mean(val_loss_list)})
    pbar.close()
    IOUs = list()
    for i, cm in enumerate(pcm):
        if i != 0:
            iou = evaluate_cm(cm, class_names[i])
            IOUs.append(iou)
    class_jaccard = torch.tensor(np.array(IOUs))

    val_miou = class_jaccard.sum().item() / class_jaccard.size(0)
    return val_miou, np.mean(val_loss_list)




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
