import os
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import ElectraTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import ElectraForSequenceClassification


class NSMCEvalDataset(Dataset):

    def __init__(self, label, input_ids, attention_mask, text_list):
        self.label = label
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.text = text_list
        self.data_size = len(label)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        label = torch.tensor(self.label[idx])
        input_ids = torch.tensor(self.input_ids[idx])
        attention_mask = torch.tensor(self.attention_mask[idx])
        text = self.text[idx]

        return label, input_ids, attention_mask, text


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--checkpoint', type=int, default=3, help='epoch of the checkpoint')
    parser.add_argument('--n_workers', type=int, default=4, help='number of epochs of training')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')

    args = parser.parse_args()
    return args


def make_eval_h5(args, file_name):

    eval_dataset = pd.read_csv(f'{args.data_root}/eval.txt', sep='\t').dropna(axis=0)  # Remove NaN
    eval_dataset.drop_duplicates(subset=['document'], inplace=True)  # Remove Duplicates

    print(f"Eval Set Size : {len(eval_dataset)}")

    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    nsmc = h5py.File(f'{args.data_root}/{file_name}.h5', 'w', rdcc_nslots=11213, rdcc_nbytes=1024 ** 3, rdcc_w0=1)

    ######################################################################################################
    ######################################### Eval Dataset ##############################################
    ######################################################################################################

    list_input_ids = []
    list_attention_mask = []
    list_label = []

    for i in tqdm(range(len(eval_dataset)), total=len(eval_dataset), desc="Parsing Eval Datasets"):
        text, label = eval_dataset.iloc[i, 1:3].values

        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=180,
            pad_to_max_length=True,
            add_special_tokens=True
        )

        input_ids = inputs['input_ids'][0]  # LongTensor / Size : 180
        attention_mask = inputs['attention_mask'][0]  # LongTensor / Size : 180

        list_input_ids.append(input_ids.numpy())
        list_attention_mask.append(attention_mask.numpy())
        list_label.append(label)

    ndarray_input_ids = np.array(list_input_ids)
    ndarray_attention_mask = np.array(list_attention_mask)
    ndarray_label = np.array(list_label)

    print("Eval Dataset Summary")
    print(f"input_ids : {ndarray_input_ids.shape} / {ndarray_input_ids.dtype}")
    print(f"attention_mask : {ndarray_attention_mask.shape} / {ndarray_attention_mask.dtype}")
    print(f"label : {ndarray_label.shape} / {ndarray_label.dtype}")

    nsmc_h5_train_grp = nsmc.create_group('eval')

    nsmc_h5_train_grp.create_dataset('input_ids',
                                     data=ndarray_input_ids,
                                     dtype=np.long)

    nsmc_h5_train_grp.create_dataset('attention_mask',
                                     data=ndarray_attention_mask,
                                     dtype=np.long)

    nsmc_h5_train_grp.create_dataset('label',
                                     data=ndarray_label,
                                     dtype=np.long)

    nsmc.close()


def eval_fine_tuned_model(args, text_list):

    nsmc = h5py.File(f'{args.data_root}/eval.h5', 'r')
    eval_dataset = nsmc['eval']

    print('\n====================== Dataset Summary ======================\n')
    print(f"Eval Label : {eval_dataset['label']}")
    print(f"Eval Input Ids : {eval_dataset['input_ids']}")
    print(f"Eval Attention Mask : {eval_dataset['attention_mask']}")
    print('\n=============================================================\n')

    eval_label = np.array(eval_dataset['label'])
    eval_input_ids = np.array(eval_dataset['input_ids'])
    eval_attention_mask = np.array(eval_dataset['attention_mask'])

    nsmc.close()

    eval_dataset = NSMCEvalDataset(eval_label, eval_input_ids, eval_attention_mask, text_list)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = nn.DataParallel(model)
    model.to(device)
    
    checkpoint = torch.load(f'logs/NSMC@01.02.2021-15:16:08/checkpoints/{args.checkpoint}')
    model.load_state_dict(checkpoint['model_state_dict'])

    train_accuracy = checkpoint['train_accuracy']
    test_accuracy = checkpoint['test_accuracy']

    ################################################
    ################## Evaluation ##################
    ################################################

    eval_correct = 0
    eval_total = 0
    num_items = 0
    model.eval()

    with open('NSMC_eval.txt', 'w') as eval_file:
        with torch.no_grad():
            for idx, (label, input_ids, attention_masks, text) in tqdm(enumerate(eval_loader), total=len(eval_loader)):

                label = label.to(device)
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)

                output = model(input_ids, attention_masks)[0]
                _, pred = torch.max(output, 1)  # values, indices

                softmax_value, _ = torch.max(F.softmax(output, dim=1), 1)

                eval_correct += (pred == label).sum()
                eval_total += len(label)

                for item in range(label.size(0)):
                    num_items += 1
                    write_eval_result(eval_file, num_items, label[item], text[item], pred[item], softmax_value[item])

            eval_accuracy = eval_correct.float() / eval_total

            print(f"Train Accuracy : {train_accuracy}")
            print(f"Test Accuracy : {test_accuracy}")
            print(f"Eval Accuracy : {eval_accuracy}")

            eval_file.write(f"Train Accuracy : {train_accuracy}\n")
            eval_file.write(f"Test Accuracy : {test_accuracy}\n")
            eval_file.write(f"Eval Accuracy : {eval_accuracy}")


def write_eval_result(eval_file, idx, label, text, prediction, value):

    if prediction == label:
        correct = 'O'
    else:
        correct = 'X'

    if label.item() == 0:
        label = '-'
    else:
        label = '+'

    if prediction.item() == 0:
        prediction = '-'
    else:
        prediction = '+'

    eval_file.write(f'[{idx}] {text}\n')
    eval_file.write(f' - Label : {label}\n')
    eval_file.write(f' - Prediction : {prediction}\n')
    eval_file.write(f' - Correct : {correct}\n')
    eval_file.write(f' - softmax(NN(x))[label_c] :{value}\n')
    eval_file.write('\n\n')


def parse_text(args):

    eval_dataset = pd.read_csv(f'{args.data_root}/eval.txt', sep='\t').dropna(axis=0)  # Remove NaN
    eval_dataset.drop_duplicates(subset=['document'], inplace=True)  # Remove Duplicates

    text_list = []
    for i in tqdm(range(len(eval_dataset)), total=len(eval_dataset), desc="Parsing Eval Datasets"):
        text, label = eval_dataset.iloc[i, 1:3].values
        text_list.append(text)

    return text_list


if __name__ == '__main__':

    args = arg_parse()

    if not os.path.isfile(f'{args.data_root}/eval.h5'):
        make_eval_h5(args, 'eval')

    text_list = parse_text(args)
    eval_fine_tuned_model(args, text_list)