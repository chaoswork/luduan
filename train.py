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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="gpt2")
    block_size: Optional[int] = field(default=1024)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, model_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=data_args.data_path, block_size=model_args.block_size)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, transformers.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = LuduanForCausalLM(LuduanConfig(block_size=model_args.block_size))
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()    

    

    
