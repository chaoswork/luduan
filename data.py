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
from datasets import load_dataset # huggingface datasets
import multiprocessing




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
        batch_size: int
           暂时放在这里实现，后续通过Trainer框架实现。
        
        """
        super(MMapDataset).__init__()

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

            total_batches = 128
            if not os.path.exists(f"{mmap_file_name}.0") or force_mmap:

                dataset = load_dataset(file_path)
                dataset = dataset['train'].map(
                    process,
                    remove_columns=['text'],
                    num_proc=num_proc
                )
                # 多进程dump

                global memmap_dump
                def memmap_dump(batch_idx):
                    batch = dataset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                    arrlen = np.sum(batch['len'], dtype=np.uint64)
                    arr = np.memmap(f"{mmap_file_name}.{batch_idx}", dtype=self.mmap_dtype,
                                    mode='w+', shape=(arrlen, ))
                    idx = 0
                    arr_batch = np.concatenate(batch['ids'])
                    assert arrlen == len(arr_batch)
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    arr.flush()

                pool = multiprocessing.Pool(num_proc)
                pool.map(memmap_dump, range(total_batches))
                dataset = None
                    

            self.arr = np.array([], dtype=self.mmap_dtype)
            print('Loading Memmap Data...')
            for i in tqdm(range(total_batches)):
                arr = np.memmap(f"{mmap_file_name}.{i}", dtype=self.mmap_dtype, mode='r')
                self.arr = np.concatenate([self.arr, arr])
            
        else:
            raise NotImplementedError
    
        print(f'Token Counts: {len(self.arr)}')



    def __iter__(self):
        """
        一个特殊的迭代器，每次随机返回一段。
        """
        while True:
           idx = torch.randint(len(self.arr) - self.block_size, (1,))
           yield torch.from_numpy((self.arr[idx:idx + self.block_size]).astype(np.int64))


    # def __len__(self):
    #     """
    #     IterableDataset并不需要__len__, 但是Dataloader需要，所以这里还是实现了。
    #     """
    #     return len(self.arr) // self.block_size
