import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import ElectraTokenizer


def preprocess(args):

    if os.path.isfile('nsmc.h5'):
        print("H5 File already exists")
    else:
        train_dataset = pd.read_csv(f'{args.data_root}/ratings_train.txt', sep='\t').dropna(axis=0)  # Remove NaN
        test_dataset = pd.read_csv(f'{args.data_root}/ratings_test.txt', sep='\t').dropna(axis=0)  # Remove NaN

        train_dataset.drop_duplicates(subset=['document'], inplace=True)  # Remove Duplicates
        test_dataset.drop_duplicates(subset=['document'], inplace=True)  # Remove Duplicates

        print(f"Train Set Size : {len(train_dataset)}")
        print(f"Test Set Size : {len(test_dataset)}")

        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

        nsmc = h5py.File(f'{args.data_root}/nsmc.h5', 'w', rdcc_nslots=11213, rdcc_nbytes=1024**3, rdcc_w0=1)

        ######################################################################################################
        ######################################### Train Dataset ##############################################
        ######################################################################################################

        list_input_ids = []
        list_attention_mask = []
        list_label = []

        for i in tqdm(range(len(train_dataset)), total=len(train_dataset), desc="Parsing Training Datasets"):

            text, label = train_dataset.iloc[i, 1:3].values

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

        print("Training Dataset Summary")
        print(f"input_ids : {ndarray_input_ids.shape} / {ndarray_input_ids.dtype}")
        print(f"attention_mask : {ndarray_attention_mask.shape} / {ndarray_attention_mask.dtype}")
        print(f"label : {ndarray_label.shape} / {ndarray_label.dtype}")

        nsmc_h5_train_grp = nsmc.create_group('train')

        nsmc_h5_train_grp.create_dataset('input_ids',
                            data=ndarray_input_ids,
                            dtype=np.long,
                            chunks=(1000, 180),
                            maxshape=(146182, 180))

        nsmc_h5_train_grp.create_dataset('attention_mask',
                            data=ndarray_attention_mask,
                            dtype=np.long,
                            chunks=(1000, 180),
                            maxshape=(146182, 180))

        nsmc_h5_train_grp.create_dataset('label',
                            data=ndarray_label,
                            dtype=np.long,
                            chunks=(1000,),  # 11 MB : Chunk Size
                            maxshape=(146182,))


        ######################################################################################################
        ######################################### Test Dataset ##############################################
        ######################################################################################################

        list_input_ids = []
        list_attention_mask = []
        list_label = []

        for i in tqdm(range(len(test_dataset)), total=len(test_dataset), desc="Parsing Test Datasets"):

            text, label = test_dataset.iloc[i, 1:3].values

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

        print("Test Dataset Summary")
        print(f"input_ids : {ndarray_input_ids.shape} / {ndarray_input_ids.dtype}")
        print(f"attention_mask : {ndarray_attention_mask.shape} / {ndarray_attention_mask.dtype}")
        print(f"label : {ndarray_label.shape} / {ndarray_label.dtype}")

        nsmc_h5_test_grp = nsmc.create_group('test')

        nsmc_h5_test_grp.create_dataset('input_ids',
                            data=ndarray_input_ids,
                            dtype=np.long,
                            chunks=(1000, 180),
                            maxshape=(49157, 180))

        nsmc_h5_test_grp.create_dataset('attention_mask',
                            data=ndarray_attention_mask,
                            dtype=np.long,
                            chunks=(1000, 180),
                            maxshape=(49157, 180))

        nsmc_h5_test_grp.create_dataset('label',
                            data=ndarray_label,
                            dtype=np.long,
                            chunks=(1000,),
                            maxshape=(49157,))
    