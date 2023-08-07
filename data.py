#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:	Huang Chao (huangchao.cpp@gmail.com)
Date: Mon Aug  7 10:15:24 2023
Brief:
1. 先实现一个nanoGPT类似的数据集合
2. 再实现一个n_chunk+shuffle的
"""
import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset # huggingface datasets




class SimpleDataset(Dataset):
    """
    最简单的实现，数据在线处理，速度慢，仅用于学习，实际不建议使用。
    """

    def __init__(self, file_path, file_type, tokenizer, block_size):
        self.block_size = block_size
        
        self.arr = []
        if file_type == 'txt':
            with open(file_path) as f:
                for line in f:
                    self.arr += tokenizer.encode(line)
        else:
            raise NotImplementedError

    def __len__(self):
        """
        Dataset 需要实现len
        """
        return len(self.arr) // self.block_size

    def __getitem__(self, idx):
        idx = torch.randint(len(self.arr) - self.block_size, (1,))
        return torch.tensor(self.arr[idx:idx + self.block_size], dtype=torch.int64)


class SimpleIterDataset(IterableDataset):
    """
    在线处理的迭代数据
    """
    def __init__(self, data_name, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        dataset = load_dataset(data_name)
        tokenized = dataset['train'].map(
            self.process_text,
            remove_columns=['text'])
        self.arr = []
        for example in tokenized:
            self.arr += example['ids']
        print(f'Token Counts: {len(self.arr)}')

    def process_text(self, example):
        ids = self.tokenizer(example['text'])
        ids.append(self.tokenizer.eos_token_id)
        return {
            'ids': ids,
            'len': len(ids)
        }

    def __iter__(self):
        idx = torch.randint(self.len(arr) - block_size, (1,))
        yield torch.from_numpy((self.arr[idx:idx + block_size]).astype(np.int64))
              


class MMapDataset(IterableDataset):
    """
    通过实现处理好的mmap二进制来加速模型训练。
    """
    def __init__(self):
        pass

    def __iter__(self):
        pass


