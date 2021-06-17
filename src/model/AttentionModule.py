import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Union


class AttentionBase(nn.Module):
    def __init__(self):
        super().__init__()
    
    def wipe_out_pad_tkn_score(self, score, lengths, dim=2):
        batch_size = score.size(0)
        mask = torch.zeros_like(score, dtype=torch.bool)
        max_len = max(lengths)
        for batch_idx in range(batch_size):
            l = lengths[batch_idx]
            if l < max_len:
                if dim == 2:
                    mask[batch_idx, :, l:] = True
                elif dim == 1:
                    mask[batch_idx, l:, :] = True
                else:
                    raise ValueError(f"`dim` in wipe_out_pad_tkn_score should be 1 or 2")
            if l == 0 and dim == 1:
                # for 0 where numbers
                mask[batch_idx, l:, :] = True
        if dim == 2:
            score = score.masked_fill(mask, -np.inf)
        elif dim == 1:
            score = score.masked_fill(mask, 0.0)
        else:
            raise ValueError(f"`dim` in wipe_out_pad_tkn_score should be 1 or 2")
        
        return score

class C2QAttention(AttentionBase):
    r"""Decoder Column to Question Attention Module"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, o_c, o_q, q_lengths, c_lengths=None, rt_attn=False):
        r"""
        Calculate for each column tokens, How much related to question tokens?
        
        o_c: LSTM output of column
        o_q: LSTM output of question 
        
        c_lengths: wipe out row length
        return context atttended to question tokens
        """
        sqrt_H = np.sqrt(o_c.size(-1)) # Apply Attention is All you Need Technique
        o_q_transform = self.linear(o_q)  # (B, T_q, H)
        score_c2q = torch.bmm(o_c, o_q_transform.transpose(1, 2)) / sqrt_H  # (B, T_c, H) x (B, H, T_q) = (B, T_c, T_q)
        score_c2q = self.wipe_out_pad_tkn_score(score_c2q, q_lengths, dim=2)
        
        prob_c2q = self.softmax(score_c2q)
        if c_lengths is not None:
            prob_c2q = self.wipe_out_pad_tkn_score(prob_c2q, c_lengths, dim=1)
        # prob_c2q: (B, T_c, T_q) -> (B, T_c, T_q, 1)
        # o_q: (B, 1, T_q, H)
        # p_col2question \odot o_q = (B, T_c, T_q, 1) \odot (B, 1, T_q, H) = (B, T_c, T_q, H)
        # -> reduce sum to T_q to get context for each column (B, T_c, H)
        context = torch.mul(prob_c2q.unsqueeze(3), o_q.unsqueeze(1)).sum(dim=2)
        if rt_attn:
            attn = prob_c2q
        else:
            attn = None
        return context, attn

class SelfAttention(AttentionBase):
    r"""Decoder Self Attention Module"""
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, o, lengths, rt_attn=False):
        r"""
        Calculate for each o tokens, How much related to o tokens?
        
        return attended summary of o
        """
        o_transform = self.linear(o)  # (B, T_o, H) -> (B, T_o, 1)
        o_transform = self.wipe_out_pad_tkn_score(o_transform, lengths) 
        o_prob = self.softmax(o_transform)  # (B, T_o, 1)
        
        o_summary = torch.mul(o, o_prob).sum(1)  # (B, T_o, H) \odot (B, T_o, 1) -> (B, H)

        if rt_attn:
            attn = o_prob
        else:
            attn = None
        return o_summary, attn