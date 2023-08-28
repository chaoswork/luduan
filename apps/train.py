#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:	Huang Chao (huangchao.cpp@gmail.com)
Date: Mon Aug 28 17:03:02 2023
Brief:
"""

import time
import torch
import numpy as np
from tqdm import tqdm

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainerCallback

from data import GlobalPointerDataset
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
    train_dataset = GlobalPointerDataset(
        file_path=data_args.data_path,
        tokenizer=tokenizer,
        max_length=model_args.block_size)

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

class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        if Y == 0:
            Y = 1e-12
        Z = len(T)
        if Z == 0:
            Z = 1e-12
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall

metrics = MetricsCalculator()


def train_step(batch_train, model, optimizer, criterion):
    # batch_input_ids:(batch_size, seq_len)    batch_labels:(batch_size, ent_type_size, seq_len, seq_len)
    # batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_train
    # batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
    #                                                                              batch_attention_mask.to(device),
    #                                                                              batch_token_type_ids.to(device),
    #                                                                              batch_labels.to(device)
    #                                                                              )
    batch_input_ids = batch_train['input_ids'].cuda()
    batch_attention_mask = batch_train['attention_mask'].cuda()
    batch_labels = batch_train['label'].cuda()

    logits = model(batch_input_ids, batch_attention_mask)

    loss = criterion(batch_labels, logits)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


#encoder = BertModel.from_pretrained(config["bert_path"])


def train(model, dataloader, epoch, optimizer):
    model.train()

    # loss func
    def loss_fun(y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = multilabel_categorical_crossentropy(y_true, y_pred)
        return loss

    # scheduler
    if True: #hyper_parameters["scheduler"] == "CAWR":
        T_mult = 1 #hyper_parameters["T_mult"]
        rewarm_epoch_num = 2 #hyper_parameters["rewarm_epoch_num"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         len(train_dataloader) * rewarm_epoch_num,
                                                                         T_mult)
    #elif hyper_parameters["scheduler"] == "Step":
    else:
        decay_rate = 0.999 #hyper_parameters["decay_rate"]
        decay_steps = 200 #hyper_parameters["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    #else:
    #    scheduler = None

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_loss = 0.
    for batch_ind, batch_data in pbar:

        loss = train_step(batch_data, model, optimizer, loss_fun)

        total_loss += loss

        avg_loss = total_loss / (batch_ind + 1)
        if scheduler is not None:
            scheduler.step()

        pbar.set_description(
            f'Project:GNER, Epoch: {epoch + 1}/"TODO:num_train_epochs", Step: {batch_ind + 1}/{len(dataloader)}')
        pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"])

        # if config["logger"] == "wandb" and batch_ind % config["log_interval"] == 0:
        #     logger.log({
        #         "epoch": epoch,
        #         "train_loss": avg_loss,
        #         "learning_rate": optimizer.param_groups[0]['lr'],
        #     })


def valid_step(batch_valid, model):
    # batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_valid
    # batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
    #                                                                              batch_attention_mask.to(device),
    #                                                                              batch_token_type_ids.to(device),
    #                                                                              batch_labels.to(device)
    #                                                                              )
    batch_input_ids = batch_valid['input_ids'].cuda()
    batch_attention_mask = batch_valid['attention_mask'].cuda()
    batch_labels = batch_valid['label'].cuda()


    with torch.no_grad():
        # logits = model(batch_input_ids, batch_attention_mask)
        logits = model(batch_input_ids, batch_attention_mask)
    sample_f1, sample_precision, sample_recall = metrics.get_evaluate_fpr(logits, batch_labels)

    return sample_f1, sample_precision, sample_recall


def valid(model, dataloader):
    model.eval()

    total_f1, total_precision, total_recall = 0., 0., 0.
    for batch_data in tqdm(dataloader, desc="Validating"):
        f1, precision, recall = valid_step(batch_data, model)

        total_f1 += f1
        total_precision += precision
        total_recall += recall

    avg_f1 = total_f1 / (len(dataloader))
    avg_precision = total_precision / (len(dataloader))
    avg_recall = total_recall / (len(dataloader))
    print("******************************************")
    print(f'avg_precision: {avg_precision}, avg_recall: {avg_recall}, avg_f1: {avg_f1}')
    print("******************************************")
    # if config["logger"] == "wandb":
    #    logger.log({"valid_precision": avg_precision, "valid_recall": avg_recall, "valid_f1": avg_f1})
    return avg_f1


if __name__ == '__main__':
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    encoder = AutoModel.from_pretrained(model_args.model_name_or_path)
    model = GlobalPointer(encoder, 25, 64)
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args, training_args=training_args)    
    

    if True: # train
        train_dataloader = DataLoader(data_module['train_dataset'],
                                      batch_size=training_args.per_device_train_batch_size)
        valid_dataloader = DataLoader(data_module['eval_dataset'],
                                          batch_size=training_args.per_device_train_batch_size)

        # optimizer
        init_learning_rate = float(training_args.learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

        max_f1 = 0.
        for epoch in range(int(training_args.num_train_epochs)):
            train(model, train_dataloader, epoch, optimizer)
            valid_f1 = valid(model, valid_dataloader)
            if valid_f1 > max_f1:
                max_f1 = valid_f1
                if valid_f1 > config["f1_2_save"]:  # save the best model
                    model_state_num = len(glob.glob(training_args.output_dir + "/model_state_dict_*.pt"))
                    torch.save(model.state_dict(),
                               os.path.join(training_args.output_dir, "model_state_dict_{}.pt".format(model_state_num)))
            print(f"Best F1: {max_f1}")
            print("******************************************")
            # if config["logger"] == "wandb":
            #     logger.log({"Best_F1": max_f1})

