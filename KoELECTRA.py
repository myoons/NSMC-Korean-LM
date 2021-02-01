import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from preprocess import preprocess

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datetime import datetime
from collections import defaultdict
from tensorboardX import SummaryWriter
from transformers import ElectraForSequenceClassification, AdamW


class NSMCDataset(Dataset):
    
    def __init__(self, label, input_ids, attention_mask):
      
        self.label = label
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.data_size = len(label)

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        label = torch.tensor(self.label[idx])
        input_ids = torch.tensor(self.input_ids[idx])
        attention_mask = torch.tensor(self.attention_mask[idx])

        return label, input_ids, attention_mask


def arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=80, help='size of the batches')
    parser.add_argument('--data_root', type=str, default='/home/myoons/PycharmProjects/pythonProject/NSMC/data', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_workers', type=int, default=8, help='number of cpu threads to use during batch generation')

    args = parser.parse_args()
    print(args)

    return args


def main(args):

    nsmc = h5py.File(f'{args.data_root}/nsmc.h5', 'r')

    train_dataset = nsmc['train']
    test_dataset = nsmc['test']

    print('\n====================== Dataset Summary ======================\n')
    print(f"Train Label : {train_dataset['label']}")
    print(f"Train Input Ids : {train_dataset['input_ids']}")
    print(f"Train Attention Mask : {train_dataset['attention_mask']}")
    print(f"Test Label : {test_dataset['label']}")
    print(f"Test Input Ids : {test_dataset['input_ids']}")
    print(f"Test Attention Mask : {test_dataset['attention_mask']}")
    print('\n=============================================================\n')

    train_label = np.array(train_dataset['label'])
    train_input_ids = np.array(train_dataset['input_ids'])
    train_attention_mask = np.array(train_dataset['attention_mask'])

    test_label = np.array(test_dataset['label'])
    test_input_ids = np.array(test_dataset['input_ids'])
    test_attention_mask = np.array(test_dataset['attention_mask'])

    nsmc.close()

    train_dataset = NSMCDataset(train_label, train_input_ids, train_attention_mask)
    test_dataset = NSMCDataset(test_label, test_input_ids, test_attention_mask)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = nn.parallel.DataParallel(model)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Plot Loss and Images in Tensorboard
    experiment_dir = 'logs/{}@{}'.format('NSMC', datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    os.makedirs(f"{experiment_dir}/checkpoints", exist_ok=True)
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    metric_dict = defaultdict(list)
    metric_dict_epoch_train = defaultdict(list)
    metric_dict_epoch_test = defaultdict(list)

    ##########################################
    ################ Training ################
    ##########################################

    n_iters_total = 0

    for epoch in range(args.n_epochs):

        total_loss_train = 0.0
        correct = 0
        total = 0
        model.train()

        for idx, (label, input_ids, attention_masks) in tqdm(enumerate(train_loader), total=len(train_loader)):

            optimizer.zero_grad()

            label = label.to(device)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)

            output = model(input_ids, attention_masks)[0]
            _, pred = torch.max(output, 1)

            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            correct += (pred == label).sum()
            total += len(label)
            train_accuracy = correct.float() / total

            if n_iters_total % 300 == 0:
                print(f"Batch Loss : {loss} / Accuracy : {train_accuracy}")

            metric_dict['train_loss'].append(loss.item())
            metric_dict['train_accuracy'].append(train_accuracy.item())
            n_iters_total += 1

            for title, value in metric_dict.items():
                writer.add_scalar('train/{}'.format(title), value[-1], n_iters_total)

        train_accuracy = correct.float() / total
        metric_dict_epoch_train['train_total_loss_epoch'].append(total_loss_train)
        metric_dict_epoch_train['train_accuracy_epoch'].append(train_accuracy)

        for title, value in metric_dict_epoch_train.items():
            writer.add_scalar('train/{}'.format(title), value[-1], epoch)

        print(f"Epoch : {epoch} / Train Loss : {total_loss_train} / Accuracy : {train_accuracy}")

        ##########################################
        ################## Test ##################
        ##########################################

        test_correct = 0
        test_total = 0
        total_loss_test = 0.0
        model.eval()

        with torch.no_grad():
            for idx, (label, input_ids, attention_masks) in tqdm(enumerate(test_loader), total=len(test_loader)):

                label = label.to(device)
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)

                output = model(input_ids, attention_masks)[0]
                _, pred = torch.max(output, 1)  # values, indices

                loss = F.cross_entropy(output, label)
                total_loss_test += loss
                test_correct += (pred == label).sum()
                test_total += len(label)

            test_accuracy = test_correct.float() / test_total
            metric_dict_epoch_test['test_total_loss_epoch'].append(total_loss_test)
            metric_dict_epoch_test['test_accuracy_epoch'].append(test_accuracy)

            for title, value in metric_dict_epoch_test.items():
                writer.add_scalar('test/{}'.format(title), value[-1], epoch)

            print(f"Test Accuracy : {test_accuracy}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
            }, os.path.join(experiment_dir, "checkpoints", str(epoch)))


if __name__ == '__main__':

    args = arg_parse()  # Setting arguments

    if not os.path.isfile(f'{args.data_root}/nsmc.h5'):
        preprocess(args)

    main(args)