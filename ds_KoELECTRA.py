import os
import h5py
import argparse
import deepspeed
import numpy as np
from tqdm import tqdm

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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


def add_argument():
    parser = argparse.ArgumentParser(description='NSMC KoELECTRA')

    # train

    parser.add_argument('-e',
                        '--epochs',
                        default=5,
                        type=int,
                        help='number of total epochs (default: 5)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():

    nsmc = h5py.File('data/nsmc.h5', 'r')

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
    
    train_loader = DataLoader(train_dataset, batch_size=55, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=55, shuffle=False, num_workers=8)

    model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    args = add_argument()

    model_engine, _, _, _ = deepspeed.initialize(args=args,
                                                model=model,
                                                model_parameters=parameters)
        

    losses = []
    accuracies = []

    for epoch in range(args.epochs):

        total_loss = 0.0
        correct = 0
        total = 0
        batches = 0

        for idx, (label, input_ids, attention_masks) in tqdm(enumerate(train_loader), total=len(train_loader)):
        
            label = label.to(model_engine.local_rank)
            input_ids = input_ids.to(model_engine.local_rank)
            attention_masks = attention_masks.to(model_engine.local_rank)

            # Model Inference
            output = model_engine(input_ids, attention_masks)[0]
            _, pred = torch.max(output, 1)
            loss = F.cross_entropy(output, label)

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            correct += (pred == label).sum()
            total += len(label)

            batches += 1

            if batches % 100 == 0:
                print(f"Batch Loss : {total_loss} / Accuracy : {correct.float() / total}")
        
        losses.append(total_loss)
        accuracies.append(correct.float() / total)
        print(f"Epoch : {epoch} / Train Loss : {total_loss} / Accuracy : {correct.float() / total}")


        test_correct = 0
        test_total = 0

        with torch.no_grad():

            for idx, (label, input_ids, attention_masks) in tqdm(enumerate(test_loader), total=len(test_loader)):

                label = label.to(model_engine.local_rank)
                input_ids = input_ids.to(model_engine.local_rank)
                attention_masks = attention_masks.to(model_engine.local_rank)

                # Model Inference
                output = model_engine(input_ids, attention_masks)[0]
                _, pred = torch.max(output, 1)

                test_correct += (pred == label).sum()
                test_total += len(label)

        print(f"Test Accuracy : {test_correct.float() / test_total}")
        model_engine.save_checkpoint('weights', epoch)

if __name__ == '__main__':

    if not os.path.isfile('data/nsmc.h5'):
        print("H5 File doesn't exists")
    else:
        main()