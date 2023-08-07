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
                    self.arr.append(tokenizer.eos_token_id)
        else:
            raise NotImplementedError
        print(f'Token Counts: {len(self.arr)}')

    def __len__(self):
        """
        Dataset 需要实现len
        """
        return len(self.arr) // self.block_size

    def __getitem__(self, idx):
        idx = torch.randint(len(self.arr) - self.block_size, (1,))
        return torch.tensor(self.arr[idx:idx + self.block_size]) #, dtype=torch.int64)


class SimpleIterDataset(IterableDataset):
    """
    在线处理的迭代数据
    """
    def __init__(self, file_path, file_type, tokenizer, block_size, num_proc=8):
        self.block_size = block_size
        
        self.arr = []
        if file_type == 'txt':
            with open(file_path) as f:
                for line in f:
                    self.arr += tokenizer.encode(line)
                    self.arr.append(tokenizer.eos_token_id)
        elif file_type == 'huggingface':
            def process(example):
                ids = tokenizer.encode(example['text']) # encode_ordinary ignores any special tokens
                ids.append(tokenizer.eos_token_id) # add the end of text token, e.g. 50256 for gpt2 bpe
                # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
                out = {'ids': ids, 'len': len(ids)}
                return out
            dataset = load_dataset(file_path)
            dataset = dataset['train'].map(
                process,
                remove_columns=['text'],
                num_proc=num_proc
            )
            for example in dataset:
                self.arr += example['ids']
            dataset = None

        else:
            raise NotImplementedError
    
        print(f'Token Counts: {len(self.arr)}')


    def __iter__(self):
        idx = torch.randint(len(self.arr) - self.block_size, (1,))
        yield torch.tensor(self.arr[idx:idx + self.block_size]) #, dtype=torch.int64)
              


class MMapDataset(IterableDataset):
    """
    通过实现处理好的mmap二进制来加速模型训练。
    """

    def __init__(self, file_path, file_type, tokenizer, block_size,
                 mmap_file_name, force_mmap=False,
                 num_proc=8):
        """
        Parameters
        ----------
        mmap_dtype : np.uint16
           注意最大值要比vocab_size要大
        """

        self.block_size = block_size
        self.mmap_dtype = np.uint32
        if tokenizer.vocab_size < 2 ** 8:
            self.mmap_dtype = np.uint8
        elif tokenizer.vocab_size < 2 ** 16:
            self.mmap_dtype = np.uint16

        if tokenizer.vocab_size >= 2 ** 32:
            raise Exception("vocab_size too large, please check it!")
        
        
        self.arr = []
        if file_type == 'txt':
            with open(file_path) as f:
                for line in f:
                    self.arr += tokenizer.encode(line)
                    self.arr.append(tokenizer.eos_token_id)
        elif file_type == 'huggingface':
            def process(example):
                ids = tokenizer.encode(example['text']) # encode_ordinary ignores any special tokens
                ids.append(tokenizer.eos_token_id) # add the end of text token, e.g. 50256 for gpt2 bpe
                # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
                out = {'ids': ids, 'len': len(ids)}
                return out

            if  not os.path.exists(mmap_file_name) or force_mmap:

                dataset = load_dataset(file_path)
                dataset = dataset['train'].map(
                    process,
                    remove_columns=['text'],
                    num_proc=num_proc
                )
                arrlen = np.sum(dataset['len'], dtype=np.uint64)
                arr = np.memmap(mmap_file_name, dtype=self.mmap_dtype,
                                mode='w+', shape=(arrlen, ))
                
                total_batches = 1024
                
                idx = 0
                for batch_idx in tqdm(range(total_batches), desc=f'writing {mmap_file_name}'):
                    # Batch together samples for faster write
                    batch = dataset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                    arr_batch = np.concatenate(batch['ids'])
                    # Write into mmap
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()

            self.arr = np.memmap(mmap_file_name, dtype=self.mmap_dtype, mode='r')
            
        else:
            raise NotImplementedError
    
        print(f'Token Counts: {len(self.arr)}')


    def __iter__(self):
        idx = torch.randint(len(self.arr) - self.block_size, (1,))
        yield torch.tensor(self.arr[idx:idx + self.block_size]) #, dtype=torch.int64)
