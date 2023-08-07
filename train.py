#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:	Huang Chao (huangchao.cpp@gmail.com)
Date: Mon Jul 17 19:37:29 2023
Brief:
"""


from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import transformers
from transformers import AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer

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



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, model_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # train_dataset = TextDataset(tokenizer=tokenizer, file_path=data_args.data_path, block_size=model_args.block_size)
    # train_dataset = SimpleIterDataset(file_path=data_args.data_path, file_type=data_args.data_type,
    #                               tokenizer=tokenizer, block_size=model_args.block_size)
    train_dataset = MMapDataset(file_path=data_args.data_path, file_type=data_args.data_type,
                                tokenizer=tokenizer, block_size=model_args.block_size,
                                mmap_file_name=data_args.mmap_file_path,
                                num_proc=40)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    

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

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    result = trainer.train()

    
    display_summary(result)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()    

    

    
