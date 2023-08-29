#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:	Huang Chao (huangchao.cpp@gmail.com)
Date: Mon Aug 28 15:28:41 2023
Brief:
"""



import time
import torch
import deepspeed
import deepspeed.comm as dist

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainerCallback

from data import GlobalPointerDataset
from data import GlobalPointerIterableDataset
from utils.train_summary import *
from ner.layers import GlobalPointer

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan-7B")
    block_size: Optional[int] = field(default=1024)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    val_data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_framework: str = field(default="pytorch")

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, model_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # train_dataset = TextDataset(tokenizer=tokenizer, file_path=data_args.data_path, block_size=model_args.block_size)
    # train_dataset = SimpleIterDataset(file_path=data_args.data_path, file_type=data_args.data_type,
    #                               tokenizer=tokenizer, block_size=model_args.block_size)
    # train_dataset = GlobalPointerIterableDataset(
    train_dataset = GlobalPointerDataset(
        file_path=data_args.data_path,
        tokenizer=tokenizer,
        max_length=model_args.block_size)

    # eval_dataset = GlobalPointerIterableDataset(
    eval_dataset = GlobalPointerDataset(
        file_path=data_args.val_data_path,
        tokenizer=tokenizer,
        max_length=model_args.block_size)
    
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=None)

def multilabel_categorical_crossentropy(y_true, y_pred):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()

# loss func
def loss_func(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


class GlobalPointerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'])
        # compute custom loss (suppose one has 3 labels with different weights)
        loss = loss_func(labels, logits)
        return (loss, outputs) if return_outputs else loss
            

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print('--------- ModelArguments ---------')
    print(model_args)
    print('--------- DataArguments ---------')
    print(data_args)
    print('--------- TrainingArguments ---------')
    print(training_args)

    base_model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = GlobalPointer(base_model, 25, 64).cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args, training_args=training_args)

    print('debug', training_args.training_framework)

    if training_args.training_framework == 'pytorch':
        trainer = GlobalPointerTrainer(
            model=model, tokenizer=tokenizer, args=training_args,
#                          callbacks=[LuduanCallback],
                          **data_module)
        result = trainer.train()
        
        display_summary(result)
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)
 
    else:
        print(f"{training_args.training_framework} is not support now. Please choose the right training framework")

                
                
                



if __name__ == "__main__":
    train()    

    

    
