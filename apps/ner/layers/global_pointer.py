#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:	Huang Chao (huangchao.cpp@gmail.com)
Date: Mon Aug 28 15:28:22 2023
Brief:
"""



import torch


def sinusoidal_position_embedding(batch_size, seq_len, output_dim, device):
    position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

    indices = torch.arange(0, output_dim // 2, dtype=torch.float)
    indices = torch.pow(10000, -2 * indices / output_dim)
    embeddings = position_ids * indices
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
    embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings


class GlobalPointer(torch.nn.Module):
    """
    https://kexue.fm/archives/8373
    """
    def __init__(self, base_model, category_heads, category_dim,
                 RoPE=True, tril_mask=True, **kwargs):
        super(GlobalPointer, self).__init__()
        self.base_model = base_model
        self.category_heads = category_heads
        self.category_dim = category_dim
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense = torch.nn.Linear(
            self.base_model.config.hidden_size,
            self.category_heads * self.category_dim * 2)
        
    def forward(self, input_ids, attention_mask) -> torch.tensor:
        """
        Parameters
        ----------
        input_ids : tensor 
            size: (B, T)
        attention_mask: tensor
            size: (B, T), 0 for mask
        token_type_ids
        Returns
        -------
        tensor
            size: 

        """

        context_outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        hidden_states = context_outputs[0]
        B, T, D = hidden_states.shape
        inputs = self.dense(hidden_states) # (B, T, cn * cd * 2)
        # (B, T, cn, cd * 2)
        inputs = inputs.view((B, T, self.category_heads, -1))
        # inputs = torch.split(inputs, self.category_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        # inputs = torch.stack(inputs, dim=-2)
        # qw, kw  (B, T, cn, cd)
        qw, kw = inputs[..., :self.category_dim], inputs[..., self.category_dim:]

        # apply_rotary_pos_emb
        if self.RoPE:
            pos_emb = sinusoidal_position_embedding(B, T, self.category_dim, qw.device)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算内积
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        # for cpm_bee
        # attention_mask = torch.zeros((B, T), device=logits.device)
        # for i, l in enumerate(length):
        #     attention_mask[i, :l] = 1

        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(B, self.category_heads, T, T)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        # scale返回
        return logits / self.category_dim ** 0.5
        
        

    
