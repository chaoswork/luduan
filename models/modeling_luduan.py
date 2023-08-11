#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:	Huang Chao (huangchao.cpp@gmail.com)
Date: Mon Jul 17 11:34:18 2023
Brief: Pytorch Luduan model, Same arch of Llama
"""


import math
import inspect
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.activations import ACT2FN

from .configuration_luduan import LuduanConfig


class RMSNorm(nn.Module):
    """ RMSNorm is equivalent to T5LayerNorm"""

    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(hidden_states.dtype)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [1, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [1, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.is_baichuan_architecture = config.is_baichuan_architecture
        if self.is_baichuan_architecture:
            self.W_pack = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        else:
            self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # output projection
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.dropout = config.dropout
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.block_size)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
##        if not False:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        #q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        if self.is_baichuan_architecture:
            q = torch.matmul(x, self.W_pack.weight[0:self.n_embd,:].T)
            k = torch.matmul(x, self.W_pack.weight[self.n_embd:2 * self.n_embd,:].T)
            v = torch.matmul(x, self.W_pack.weight[2 * self.n_embd:,:].T)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        cos, sin = self.rotary_emb(v, seq_len=k.shape[-2])
        # position_ids暂时没有座位参数传进来，后续可以改进。
        position_ids = torch.arange(0, T).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
#        if False:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # llama mask_fill用的是dtype的最小值。
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, torch.finfo(x.dtype).min)
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.o_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.act_fn    = ACT2FN[config.hidden_act]


    def forward(self, x):
        x_gate = self.act_fn(self.gate_proj(x))
        x_up = self.up_proj(x)
        return self.down_proj(x_gate * x_up)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x



class PreTrainedModel(PreTrainedModel):
    # 确保自动加载的时候能够找到对应的Config
    config_class = LuduanConfig

    def _init_weights(self, module):
        """
        初始化权重最好放在这里，torch的老版本需要。
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Model(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        # self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params(non_embedding=False)/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params    
    


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None):
#         print('debug', input_ids.shape)

        idx = input_ids
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        x = self.embed_tokens(idx) # token embeddings of shape (b, t, n_embd)

        # no drop
        # x = self.transformer.drop(tok_emb + pos_emb)
        all_hidden_states = () if output_hidden_states else None
        for block in self.layers:
            x = block(x)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
        x = self.norm(x)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)


        # TODO: implement in the future
        next_cache = None
        all_self_attns = None

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



class LuduanForCausalLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # save的时候保存auto_map字段，用于使用Auto来加载
        self.config.auto_map = {
            "AutoConfig": "configuration_luduan.LuduanConfig",
            "AutoModelForCausalLM": "modeling_luduan.LuduanForCausalLM"
        }
        self.model = Model(config)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head.weight = self.model.transformer.wte.weight

        # Initialize weights and apply final processing
        self.post_init()
        

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,)

        x = outputs[0]
        if labels is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # 这里默认给的labels没有shift
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # print(shift_logits.view(-1, shift_logits.size(-1)).shape)
            # print(shift_labels.view(-1).shape)
            
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.model.embed_tokens.weight.numel()
        return n_params    
    

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        """
        改函数保证可以使用.generate函数
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def load_weights_from_baichuan(self):
        """
        已废弃，仅供学习使用
        由于baichuan的attention实现和llama略有不同，所以目前通过copy state_dict的方式实现,这就导致
        显存占用提升了一倍。
        pytorch的model是pickle格式的，虽然结构上通过很多ID来进行分割，但是如何只加载部分参数还没有
        很方便的方法，只能修改torch的load函数。
        baichuan的W_pack就相当于q_proj, k_proj, v_proj拼接起来。
        通过Attention 引入is_baichuan_architecture完美解决了这个问题。
        """

        baichuan = AutoModelForCausalLM.from_pretrained('baichuan-inc/Baichuan-7B',trust_remote_code=True).to('cuda:0')

        for weight_name in luduan.state_dict():

            if weight_name not in baichuan.state_dict():
                if weight_name.endswith('self_attn.bias'):
                    continue
                layer_no = weight_name.split('.')[2]
                w_name = f'model.layers.{layer_no}.self_attn.W_pack.weight'
                # print(baichuan.state_dict()[w_name].shape)
                if 'q_proj' in weight_name:
                    luduan.state_dict()[weight_name].copy_(baichuan.state_dict()[w_name][0:luduan.config.n_embd,:])
                elif 'k_proj' in weight_name:
                    luduan.state_dict()[weight_name].copy_(baichuan.state_dict()[w_name][luduan.config.n_embd:2*luduan.config.n_embd,:])
                elif 'v_proj' in weight_name:
                    luduan.state_dict()[weight_name].copy_(baichuan.state_dict()[w_name][2*luduan.config.n_embd:,:])

                # print(weight_name)
            else:
                luduan.state_dict()[weight_name].copy_(baichuan.state_dict()[weight_name]    )
        # 释放显存
        baichuan = None
        torch.cuda.empty_cache()


    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

