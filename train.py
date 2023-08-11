#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:	Huang Chao (huangchao.cpp@gmail.com)
Date: Mon Jul 17 19:37:29 2023
Brief:
"""

import time
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainerCallback

from models.modeling_luduan import LuduanForCausalLM
from models.configuration_luduan import LuduanConfig

from data import SimpleDataset, SimpleIterDataset, MMapDataset
from utils.train_summary import *

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan-7B")
    block_size: Optional[int] = field(default=1024)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    mmap_file_path: str = field(default='./train.bin', metadata={"help": "Path to the mmap data."})
    data_type: str = field(default='txt', metadata={"help": "training data type."})



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, model_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # train_dataset = TextDataset(tokenizer=tokenizer, file_path=data_args.data_path, block_size=model_args.block_size)
    # train_dataset = SimpleIterDataset(file_path=data_args.data_path, file_type=data_args.data_type,
    #                               tokenizer=tokenizer, block_size=model_args.block_size)
    train_dataset = MMapDataset(file_path=data_args.data_path, file_type=data_args.data_type,
                                tokenizer=tokenizer, block_size=model_args.block_size,
                                mmap_file_name=data_args.mmap_file_path, force_mmap=False,
                                num_proc=40)
    # 加上下面这一句会报错，待排查
    # 其实不用加下面这一句，batch_size通过参数会传给trainer
    # train_dataset = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size)
    
    # 需要注意DataCollatorForLanguageModeling的input_ids和labels是完全一样的，如果使用它，则需要再gpt/llama的内部实现labels的位移
    # https://discuss.huggingface.co/t/where-does-the-transformers-do-the-target-text-shifting-in-causal-lm/32408
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    ### Debug
    # samples = []
    # for i in range(5):
    #     samples.append(next(train_dataset)[0])
    # print(samples)
    # out = data_collator(samples)
    # for key in out:
    #     print(f"{key} shape: {out[key].shape}")
    #     print(f"{key}:", out[key])
    # end debug
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class LuduanCallback(TrainerCallback):
    def __init__(self):
        self.create_time = time.time()
        self.last_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            print(logs)
            # print('args=', args)
            # print('state=', state)
            # print('control=', control)
            # print('kwargs=', kwargs)
            cur_time = time.time()
            time_used = cur_time - self.last_time
            self.last_time = cur_time
            # print('logging_step', args.logging_steps)
            # print('batch_size', args.per_device_train_batch_size)
            mfu = kwargs['model'].estimate_mfu(
                args.logging_steps * args.per_device_train_batch_size,
                time_used)
            print(f'{args.logging_steps} Steps Time Used: {time_used}s')
            print(f'Estimate MFU:\t{100 * mfu}%')
            # 312e12 is only for A100
            print(f"Torch Given MFU:\t{100 * state.total_flos / (312e12 * (cur_time - self.create_time))}%")
            

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, transformers.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print('--------- ModelArguments ---------')
    print(model_args)
    print('--------- DataArguments ---------')
    print(data_args)
    print('--------- TrainingArguments ---------')
    print(training_args)

    nano_luduan_config = LuduanConfig(
        vocab_size=64000,
        n_embd=768,
        n_layer=12,
        n_head=12,
        block_size=1024,
        intermediate_size=768 * 4)

    model = LuduanForCausalLM(nano_luduan_config)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args, training_args=training_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args,
                      callbacks=[LuduanCallback],
                      **data_module)
    result = trainer.train()

    
    display_summary(result)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()    

    

    
