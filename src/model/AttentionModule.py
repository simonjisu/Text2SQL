import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Union

# Attention Layers
class AttentionBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def wipe_out_pad_tkn_score(self, score: torch.Tensor, lengths: List[int], dim: int) -> torch.Tensor:
        """Wipe out the unnesscary pad tokens in `score`

        Args:
            score (torch.Tensor): [description]
            lengths (List[int]): [description]
            dim (int): [description]

        Raises:
            ValueError: [description]

        Returns:
            torch.Tensor: [description]
        """     
        max_len = max(lengths)
        for batch_idx, length in enumerate(lengths):
            if length < max_len:
                if dim == 2:
                    score[batch_idx, :, length:] = -10000000
                elif dim == 1:
                    score[batch_idx, length:, :] = 0.0
                else:
                    raise ValueError(f"`dim` in wipe_out_pad_tkn_score should be 1 or 2")
        return score 


class C2QAttention(AttentionBase):
    r"""Decoder Column to Question Attention Module"""
    def __init__(self, in_features: int, out_features: int) -> None:        
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, o_c: torch.Tensor, o_q: torch.Tensor, q_lengths: List[int], c_lengths: Union[List[int], None]=None, rt_attn: bool=False) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:        
        """Calculate for each column tokens, How much related to question tokens?

        Args:
            o_c (torch.Tensor): LSTM output of columns
            o_q (torch.Tensor): LSTM output of questions 
            q_lengths (List[int]): [description]
            c_lengths (Union[List[int], None], optional): if not `None`, it should wipe out row in the score. Defaults to None.
            rt_attn (bool, optional): [description]. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, None]]: context atttended from question and attention matrix
        """        
        sqrt_H = torch.sqrt(torch.FloatTensor([o_c.size(-1)], device=o_c.device))  # Apply Attention is All you Need Technique
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
    def __init__(self, in_features: int, out_features: int=1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, o: torch.Tensor, lengths: List[int], rt_attn: bool=False) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Calculate for each `o` tokens, How much related to `o` tokens?

        Args:
            o (torch.Tensor): [description]
            lengths (List[int]): [description]
            rt_attn (bool, optional): [description]. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, None]]: attended summary of o
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
