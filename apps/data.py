#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:	Huang Chao (huangchao.cpp@gmail.com)
Date: Mon Aug  7 10:15:24 2023
Brief:
1. 先实现一个nanoGPT类似的数据集合
2. 再实现一个n_chunk+shuffle的
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, IterableDataset
import multiprocessing
import json




class GlobalPointerDataset(Dataset):
    """
    Just a simple dataset
    """

    def __init__(self, file_path, tokenizer, max_length):

        self.tokenizer = tokenizer
        self.data = []
        self.label_ids = {
            "人名": 0, "地名": 1, "文学影视艺术作品": 2, "游戏": 3, "公司/组织": 4,
            "品牌": 5, "商品": 6, "历史": 7, "天文/地理": 8, "社会": 9, "法律": 10,
            "经济/金融": 11, "管理学": 12, "计算机/互联网": 13, "医学": 14,
            "心理学": 15, "军事": 16, "体育": 17, "数学": 18, "物理": 19, "化学": 20,
            "生物": 21, "政治": 22, "哲学": 23, "其他": 24
        }
        self.max_length = max_length
        with open(file_path) as f:
            for line in f:
                example = json.loads(line)


                self.data.append(example)
                
        print(f"Dataset:\t{file_path} loaded")
                    
    def __len__(self):
        """
        Dataset 需要实现len
        """
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        tokenized_info = self.tokenizer(
            example["input"],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=False,
        )

        # make labels
        label_matrix = torch.zeros((len(self.label_ids), self.max_length, self.max_length))

        length = sum(tokenized_info['attention_mask'])
        for label_info in sorted(example['entities'], key=lambda x: x['start_offset']):
            if label_info['label'] not in self.label_ids:
                continue
            label_id = self.label_ids[label_info['label']]
            
            entity = example['input'][label_info['start_offset']: label_info['end_offset']]
            # print('debug', entity, label_info['label'])
            entity_ids = self.tokenizer(entity, add_special_tokens=False)['input_ids']
            for i in range(length - len(entity_ids)):
                # print(type(input_ids), type(entity_ids))
                # TODO 先用N^2方法实现，需要的话改为kmp
                if tokenized_info['input_ids'][i: i + len(entity_ids)] == entity_ids:
                    # print('new_start_offset:', i)
                    # print('new_end_offset:', i + len(entity_ids))
                    label_matrix[label_id, i, i + len(entity_ids)] = 1
        
        return {
            "input_ids": torch.tensor(tokenized_info['input_ids']),
            "attention_mask": torch.tensor(tokenized_info['attention_mask']),
            "label": label_matrix
        }



class GlobalPointerIterableDataset(IterableDataset):
    """
    Just a iterable dataset
    """

    def __init__(self, file_path, tokenizer, max_length):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.label_ids = {
            "人名": 0, "地名": 1, "文学影视艺术作品": 2, "游戏": 3, "公司/组织": 4,
            "品牌": 5, "商品": 6, "历史": 7, "天文/地理": 8, "社会": 9, "法律": 10,
            "经济/金融": 11, "管理学": 12, "计算机/互联网": 13, "医学": 14,
            "心理学": 15, "军事": 16, "体育": 17, "数学": 18, "物理": 19, "化学": 20,
            "生物": 21, "政治": 22, "哲学": 23, "其他": 24
        }
        self.max_length = max_length
        self.file_path = file_path
                

    def __iter__(self):
        with open(self.file_path) as f:
            for line in f:
                example = json.loads(line)

                tokenized_info = self.tokenizer(
                    example["input"],
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                )
                # make labels
                label_matrix = torch.zeros((len(self.label_ids), self.max_length, self.max_length))

                length = sum(tokenized_info['attention_mask'])
                for label_info in sorted(example['entities'], key=lambda x: x['start_offset']):
                    if label_info['label'] not in self.label_ids:
                        continue
                    label_id = self.label_ids[label_info['label']]
                    
                    entity = example['input'][label_info['start_offset']: label_info['end_offset']]
                    # print('debug', entity, label_info['label'])
                    entity_ids = self.tokenizer(entity)['input_ids']
                    for i in range(length - len(entity_ids)):
                        # print(type(input_ids), type(entity_ids))
                        # TODO 先用N^2方法实现，需要的话改为kmp
                        if tokenized_info['input_ids'][i: i + len(entity_ids)] == entity_ids:
                            # print('new_start_offset:', i)
                            # print('new_end_offset:', i + len(entity_ids))
                            label_matrix[label_id, i, i + len(entity_ids)] = 1
                yield {
                    "input_ids": torch.tensor(tokenized_info['input_ids']),
                    "attention_mask": torch.tensor(tokenized_info['attention_mask']),
                    "label": label_matrix
                }


                    
