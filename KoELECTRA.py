import os
import h5py
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


if not os.path.isfile('data/nsmc.h5'):
    print("H5 File doesn't exists")
else:

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

  train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")
  model = nn.parallel.DataParallel(model)
  model.to(device)

  optimizer = AdamW(model.parameters(), lr=1e-5)

  epochs = 5
  losses = []
  accuracies = []

  for epoch in range(epochs):

    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    model.train()

    for idx, (label, input_ids, attention_masks) in tqdm(enumerate(train_loader), total=len(train_loader)):

      optimizer.zero_grad()
    
      label = label.to(device)
      input_ids = input_ids.to(device)
      attention_masks = attention_masks.to(device)

      output = model(input_ids, attention_masks)[0]

      loss = F.cross_entropy(output, label)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

      _, pred = torch.max(output, 1)

      correct += (pred == label).sum()
      total += len(label)

      batches += 1
      if batches % 100 == 0:
        print(f"Batch Loss : {total_loss} / Accuracy : {correct.float() / total}")
    
    losses.append(total_loss)
    accuracies.append(correct.float() / total)
    print(f"Epoch : {epoch} / Train Loss : {total_loss} / Accuracy : {correct.float() / total}")

    model.eval()

    test_correct = 0
    test_total = 0

    for idx, (label, input_ids, attention_masks) in tqdm(enumerate(test_loader), total=len(test_loader)):

      label = label.to(device)
      input_ids = input_ids.to(device)
      attention_masks = attention_masks.to(device)

      output = model(input_ids, attention_masks)[0]

      _, pred = torch.max(output, 1)

      test_correct += (pred == label).sum()
      test_total += len(label)

    print(f"Test Accuracy : {test_correct.float() / test_total}")

    torch.save(model.state_dict(), f"model_{epoch}.pt")