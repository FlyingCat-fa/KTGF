import glob
import logging
import os
import pickle
import numpy as np
import copy

import torch
from torch.utils.data import Dataset

from utils import dist_utils
from .common_utils import read_json


logger = logging.getLogger()

class ReaderMedDataset_gen(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer, stage='stage1', stage1_index_file=None):
        # stage='stage1','stage2'
        self.tokenizer = tokenizer
        self.stage = stage
        if dist_utils.is_local_master():
            logger.info(f"Data dir: {data_dir}")
        data = read_json(path=data_dir)
        self.data = []
        if self.stage == 'stage1':
            for example in data:
                if example["term_id"] == -1:
                    self.data.append(example)
        elif self.stage == 'stage2':
            if stage1_index_file is None:
                for example in data:
                    if example["term_id"] != -1:
                        self.data.append(example)
            else:
                stage1_index = read_json(stage1_index_file)
                for example in data:
                    if [example['dial_id'], example['window_id'], example['term_id']] in stage1_index:
                        self.data.append(example)

        if dist_utils.is_local_master():
            logger.info(f"Total data size: {len(self.data)}")

        # self.data = data

    def convert_example_to_feature(
        self,
        example,
        max_input_len=512,
        max_output_len=512,
    ):
        input = self.tokenizer.encode(example['context'].lower(), add_special_tokens=True)[:max_input_len]
        input_mask = [1] * len(input)
        label = self.tokenizer.encode(example['output'].lower())[1:max_output_len]
        dial_id = example['dial_id']
        window_id = example['window_id']
        term_id = example['term_id']
        

        return (input, input_mask, label, dial_id,  window_id, term_id)

    def __getitem__(self, idx):
        return self.convert_example_to_feature(self.data[idx])

    def __len__(self):
        return len(self.data)

class ReaderMedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer):
        self.tokenizer = tokenizer
        if dist_utils.is_local_master():
            logger.info(f"Data dir: {data_dir}")
        data = read_json(path=data_dir)

        if dist_utils.is_local_master():
            logger.info(f"Total data size: {len(data)}")

        self.data = data

    def convert_example_to_feature(
        self,
        example,
        max_input_len=512,
        max_output_len=512,
    ):
        input = self.tokenizer.encode(example['context'].lower(), add_special_tokens=True)[:max_input_len]
        input_mask = [1] * len(input)
        label = self.tokenizer.encode(example['output'].lower())[1:max_output_len]
        dial_id = example['dial_id']
        window_id = example['window_id']
        term_id = example['term_id']
        

        return (input, input_mask, label, dial_id,  window_id, term_id)

    def __getitem__(self, idx):
        return self.convert_example_to_feature(self.data[idx])

    def __len__(self):
        return len(self.data)


class ReaderDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        paths = glob.glob(os.path.join(data_dir, '*'))

        if dist_utils.is_local_master():
            logger.info(f"Data dir: {data_dir}")
            logger.info(f"Data paths: {paths}")

        assert paths, "No Data files found."
        data = []
        for path in paths:
            with open(path, "rb") as f:
                data.extend(pickle.load(f))

        if dist_utils.is_local_master():
            logger.info(f"Total data size: {len(data)}")

        for d in data:
            d.to_tensor()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

