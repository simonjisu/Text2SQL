import torch
import torch.nn as nn
from AttentionModule import C2QAttention, SelfAttention
from typing import List, Dict, Any, Tuple, Union


class SelectDecoder(nn.Module):
    r"""SELECT Decoder"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int=2, dropout_ratio:float=0.3) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        
        self.lstm_q = nn.LSTM(input_size, int(hidden_size / 2), num_layers, dropout=dropout_ratio, batch_first=True, bidirectional=True)
        self.lstm_h = nn.LSTM(input_size, int(hidden_size / 2), num_layers, dropout=dropout_ratio, batch_first=True, bidirectional=True)
        
        self.col_context_linear = nn.Linear(2*hidden_size, hidden_size)
        self.col2question_attn = C2QAttention(hidden_size, hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.Tanh(),
            nn.Linear(2*hidden_size, output_size)
        )

    def forward(self, question_padded: torch.Tensor, header_padded: torch.Tensor, col_padded: torch.Tensor, question_lengths: List[int], col_lengths: List[int], rt_attn=False):
        r"""
        predict column index
        """
        batch_size, n_col, _ = col_padded.size()
        o_q, (h_q, c_q) = self.lstm_q(question_padded)  # o_q: (B, T_q, H)
        o_c, (h_c, c_c) = self.lstm_h(col_padded)  # o_c: (B, T_c, H)
        o_h, (h_h, c_h) = self.lstm_h(header_padded)  # h_h: (n_direc*num_layers, B, H/2)
        
        header_summary = torch.cat([h for h in h_h[-2:]], dim=1).unsqueeze(1).repeat(1, n_col, 1)  # (B, T_c, H)
        col_context = torch.cat([o_c, header_summary], dim=2)  # (B, T_c, 2H)
        col_context = self.col_context_linear(col_context)  # (B, T_c, H)
        col_q_context, attn = self.col2question_attn(col_context, o_q, question_lengths, col_lengths, rt_attn)  # (B, T_c, H), (B, T_c, T_q)
        
        vec = torch.cat([col_q_context, col_context], dim=2)  # (B, T_c, 2H)
        output = self.output_layer(vec)
        # TODO: add penalty for padded header(column) information
        
        return output.squeeze(-1), attn
    

class AggDecoder(nn.Module):
    r"""AGG Decoder"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int=2, dropout_ratio:float=0.3) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        
        self.lstm_q = nn.LSTM(input_size, int(hidden_size / 2), num_layers, dropout=dropout_ratio, batch_first=True, bidirectional=True)
        self.lstm_h = nn.LSTM(input_size, int(hidden_size / 2), num_layers, dropout=dropout_ratio, batch_first=True, bidirectional=True)
        
        self.col_context_linear = nn.Linear(2*hidden_size, hidden_size)
        self.col2question_attn = C2QAttention(hidden_size, hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
                
    def forward(self, question_padded: torch.Tensor, header_padded: torch.Tensor, col_padded: torch.Tensor, question_lengths: List[int], col_lengths: List[int], select_idxes: List[int], rt_attn=False):
        r"""
        predict agg index
        select_prob: selected argmax indices of select_output score
        """
        batch_size, n_col, _ = col_padded.size()
        o_q, (h_q, c_q) = self.lstm_q(question_padded)  # o_q: (B, T_q, H)
        o_c, (h_c, c_c) = self.lstm_h(col_padded)  # o_c: (B, T_c, H)
        o_h, (h_h, c_h) = self.lstm_h(header_padded)  # h_h: (n_direc*num_layers, B, H/2)
        
        header_summary = torch.cat([h for h in h_h[-2:]], dim=1).unsqueeze(1).repeat(1, n_col, 1)  # (B, T_c, H)
        col_context = torch.cat([o_c, header_summary], dim=2)  # (B, T_c, 2H)
        col_context = self.col_context_linear(col_context)  # (B, T_c, H)
        
        col_selected = col_context[list(range(batch_size)), select_idxes].unsqueeze(1)  # col_selected: (B, 1, H)
        
        col_q_context, attn = self.col2question_attn(col_selected, o_q, question_lengths, col_lengths, rt_attn)  # (B, 1, H), (B, 1, T_q)
        output = self.output_layer(col_q_context.squeeze(1))
        
        return output, attn
    
    
class WhereNumDecoder(nn.Module):
    r"""WHERE number Decoder"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int=2, dropout_ratio:float=0.3, max_where_conds=4) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.max_where_conds = max_where_conds
        if self.output_size > self.max_where_conds+1:
            # HERE output will be dilivered to cross-entropy loss, not guessing the real number of where clause
            raise ValueError(f"`WhereNumDecoder` only support maximum {max_where_conds} where clause")
        
        self.lstm_q = nn.LSTM(input_size, int(hidden_size / 2), num_layers, dropout=dropout_ratio, batch_first=True, bidirectional=True)
        self.lstm_h = nn.LSTM(input_size, int(hidden_size / 2), num_layers, dropout=dropout_ratio, batch_first=True, bidirectional=True)
        
        self.col_self_attn = SelfAttention(2*hidden_size, 1)
        self.lstm_q_hidden_init_linear = nn.Linear(2*hidden_size, 2*hidden_size)
        self.lstm_q_cell_init_linear = nn.Linear(2*hidden_size, 2*hidden_size)
        
        self.context_self_attn = SelfAttention(hidden_size, 1)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        
        
    def forward(self, question_padded: torch.Tensor, header_padded: torch.Tensor, col_padded: torch.Tensor, question_lengths: List[int], col_lengths: List[int], rt_attn=False):
        r"""
        predict agg index
        select_prob: selected argmax indices of select_output score
        """
        batch_size, n_col, _ = col_padded.size()
        o_c, (h_c, c_c) = self.lstm_h(col_padded)  # o_c: (B, T_c, H)
        o_h, (h_h, c_h) = self.lstm_h(header_padded)  # h_h: (n_direc*num_layers, B, H/2)
        
        header_summary = torch.cat([h for h in h_h[-2:]], dim=1).unsqueeze(1).repeat(1, n_col, 1)  # (B, T_c, H)
        col_context = torch.cat([o_c, header_summary], dim=2)  # (B, T_c, 2H)

        col_self_attn, col_attn = self.col_self_attn(col_context, col_lengths, rt_attn)  # (B, 2H), (B, T_c)

        h_0 = self.lstm_q_hidden_init_linear(col_self_attn)  # (B, 2H)
        h_0 = h_0.view(batch_size, 2*self.num_layers, -1).transpose(0, 1).contiguous()  # (B, n_direc*num_layers, H/2) -> (n_direc*num_layers, B, H/2)
        c_0 = self.lstm_q_cell_init_linear(col_self_attn)  # (B, 2H)
        c_0 = c_0.view(batch_size, 2*self.num_layers, -1).transpose(0, 1).contiguous()  # (B, n_direc*num_layers, H/2) -> (n_direc*num_layers, B, H/2)
        
        o_q, (h_q, c_q) = self.lstm_q(question_padded, (h_0, c_0))  # o_q: (B, T_q, H)
        o_summary, o_attn = self.context_self_attn(o_q, question_lengths, rt_attn)  # (B, H), (B, T_q)
        output = self.output_layer(o_summary)
        
        return output, (col_attn, o_attn)

    
class WhereColumnDecoder(nn.Module):
    r"""WHERE Column Decoder"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int=1, num_layers: int=2, dropout_ratio:float=0.3, max_where_conds: int=4) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio

        self.lstm_q = nn.LSTM(input_size, int(hidden_size / 2), num_layers, dropout=dropout_ratio, batch_first=True, bidirectional=True)
        self.lstm_h = nn.LSTM(input_size, int(hidden_size / 2), num_layers, dropout=dropout_ratio, batch_first=True, bidirectional=True)
        
        self.col_context_linear = nn.Linear(2*hidden_size, hidden_size)
        self.col2question_attn = C2QAttention(hidden_size, hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.Tanh(),
            nn.Linear(2*hidden_size, output_size)
        )

    def forward(self, question_padded: torch.Tensor, header_padded: torch.Tensor, col_padded: torch.Tensor, question_lengths: List[int], col_lengths: List[int], rt_attn=False):
        r"""
        predict column index
        """
        batch_size, n_col, _ = col_padded.size()
        o_q, (h_q, c_q) = self.lstm_q(question_padded)  # o_q: (B, T_q, H)
        o_c, (h_c, c_c) = self.lstm_h(col_padded)  # o_c: (B, T_c, H)
        o_h, (h_h, c_h) = self.lstm_h(header_padded)  # h_h: (n_direc*num_layers, B, H/2)
        
        header_summary = torch.cat([h for h in h_h[-2:]], dim=1).unsqueeze(1).repeat(1, n_col, 1)  # (B, T_c, H)
        col_context = torch.cat([o_c, header_summary], dim=2)  # (B, T_c, 2H)
        col_context = self.col_context_linear(col_context)  # (B, T_c, H)
        col_q_context, attn = self.col2question_attn(col_context, o_q, question_lengths, col_lengths, rt_attn)  # (B, T_c, H), (B, T_c, T_q)
        
        vec = torch.cat([col_q_context, col_context], dim=2)  # (B, T_c, 2H)
        output = self.output_layer(vec)
        # TODO: add penalty for padded header(column) information
        
        return output.squeeze(-1), attn
    
    
class WhereOpDecoder(nn.Module):
    r"""WHERE Opperator Decoder"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int=2, dropout_ratio: float=0.3, max_where_conds: int=4) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.max_where_conds = max_where_conds
        
        self.lstm_q = nn.LSTM(input_size, int(hidden_size / 2), num_layers, dropout=dropout_ratio, batch_first=True, bidirectional=True)
        self.lstm_h = nn.LSTM(input_size, int(hidden_size / 2), num_layers, dropout=dropout_ratio, batch_first=True, bidirectional=True)
        
        self.col_context_linear = nn.Linear(2*hidden_size, hidden_size)
        self.col2question_attn = C2QAttention(hidden_size, hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.Tanh(),
            nn.Linear(2*hidden_size, output_size)
        )
    
    def forward(self, question_padded: torch.Tensor, header_padded: torch.Tensor, col_padded: torch.Tensor, question_lengths: List[int], where_nums: List[int], where_col_idxes: List[List[int]], rt_attn=False):
        r"""
        predict agg index
        select_prob: selected argmax indices of select_output score
        max_where_col_nums is settled at WhereColumnDecoder, but it can be lower than or equal to `max_where_conds`
        """
        batch_size, n_col, _ = col_padded.size()
        o_q, (h_q, c_q) = self.lstm_q(question_padded)  # o_q: (B, T_q, H)
        o_c, (h_c, c_c) = self.lstm_h(col_padded)  # o_c: (B, T_c, H)
        o_h, (h_h, c_h) = self.lstm_h(header_padded)  # h_h: (n_direc*num_layers, B, H/2)
        
        header_summary = torch.cat([h for h in h_h[-2:]], dim=1).unsqueeze(1).repeat(1, n_col, 1)  # (B, T_c, H)
        col_context = torch.cat([o_c, header_summary], dim=2)  # (B, T_c, 2H)
        col_context = self.col_context_linear(col_context)  # (B, T_c, H)
        col_context_padded = self.get_context_padded(col_context, where_nums, where_col_idxes)  # (B, max_where_col_nums, H)
        
        col_q_context, attn = self.col2question_attn(col_context_padded, o_q, question_lengths, where_nums, rt_attn)  # (B, max_where_col_nums, H), (B, max_where_col_nums, T_q)
        
        vec = torch.cat([col_q_context, col_context_padded], dim=2)  # (B, max_where_col_nums, 2H)
        output = self.output_layer(vec)  # (B, max_where_col_nums, n_cond_ops)
        # TODO: add penalty for padded header(column) information
        return output
        
    def get_context_padded(self, col_context, where_nums, where_col_idxes):
        r"""
        Select the where column index and pad if some batch doesn't match the max length of tensor
        In case for have different where column lengths
        """
        batch_size, n_col, hidden_size = col_context.size()
        max_where_col_nums = max(where_nums)
        batches = [col_context[i, batch_col] for i, batch_col in enumerate(where_col_idxes)]  # [(where_col_nums, hidden_size), ...]  len = B
        batches_padded = []
        for b in batches:
            where_col_nums = b.size(0)
            if where_col_nums < max_where_col_nums:
                b_padded = torch.cat([b, torch.zeros((max_where_col_nums-where_col_nums), hidden_size, device=col_context.device)], dim=0)
            else:
                b_padded = b
            batches_padded.append(b_padded)  # (max_where_col_nums, hidden_size)
            
        return torch.stack(batches_padded) # (B, max_where_col_nums, hidden_size)
    
    
class WhereValueDecoder(nn.Module):
    r"""WHERE Value Decoder"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int=2, dropout_ratio: float=0.3, max_where_conds: int=4, n_cond_ops: int=4,
                 start_tkn_id: int=8002, end_tkn_id: int=8003, embedding_layer: torch.nn.modules.sparse.Embedding=None) -> None:                 
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.max_where_conds = max_where_conds
        self.n_cond_ops = n_cond_ops
        
        self.start_tkn_id = start_tkn_id
        self.end_tkn_id = end_tkn_id
        
        self.lstm_q = nn.LSTM(input_size, int(hidden_size / 2), num_layers, dropout=dropout_ratio, batch_first=True, bidirectional=True)
        self.lstm_h = nn.LSTM(input_size, int(hidden_size / 2), num_layers, dropout=dropout_ratio, batch_first=True, bidirectional=True)
        
        self.col_context_linear = nn.Linear(2*hidden_size, hidden_size)
        self.where_op_linear = nn.Linear(n_cond_ops, hidden_size)
        self.col2question_attn = C2QAttention(hidden_size, hidden_size)
        if embedding_layer is None:
            raise KeyError("Must initialize the embedding_layer to BertModel's word embedding layer")
        else:
            self.embedding_layer = embedding_layer
            vocab_size, bert_hidden_size = embedding_layer.weight.data.size()
            self.output_lstm_hidden_init_linear = nn.Linear(3*hidden_size, bert_hidden_size)
            self.output_lstm_cell_init_linear = nn.Linear(3*hidden_size, bert_hidden_size)
            self.output_lstm = nn.LSTM(bert_hidden_size, bert_hidden_size, 1, batch_first=True)
            self.output_linear = nn.Linear(bert_hidden_size, vocab_size)
            self.output_linear.weight.data = embedding_layer.weight.data

    def forward(self, question_padded: torch.Tensor, header_padded: torch.Tensor, col_padded: torch.Tensor, question_lengths: List[int], where_nums: List[int], where_col_idxes: List[List[int]], where_op_idxes: List[List[int]], value_tkn_max_len=None, g_wv_tkns=None, rt_attn=False):
        r"""
        predict agg index
        select_prob: selected argmax indices of select_output score
        max_where_col_nums is setted at WhereColumnDecoder
        value_tkn_max_len = Test if None else Train
        g_wv_tkns = When Train should not be None
        
        """
        batch_size, n_col, _ = col_padded.size()
        o_q, (h_q, c_q) = self.lstm_q(question_padded)  # o_q: (B, T_q, H)
        o_c, (h_c, c_c) = self.lstm_h(col_padded)  # o_c: (B, T_c, H)
        o_h, (h_h, c_h) = self.lstm_h(header_padded)  # h_h: (n_direc*num_layers, B, H/2)
        
        header_summary = torch.cat([h for h in h_h[-2:]], dim=1).unsqueeze(1).repeat(1, n_col, 1)  # (B, T_c, H)
        col_context = torch.cat([o_c, header_summary], dim=2)  # (B, T_c, 2H)
        col_context = self.col_context_linear(col_context)  # (B, T_c, H)
        col_context_padded = self.get_context_padded(col_context, where_nums, where_col_idxes)  # (B, max_where_col_nums, H)
        
        col_q_context, attn = self.col2question_attn(col_context_padded, o_q, question_lengths, where_nums, rt_attn)  # (B, max_where_col_nums, H), (B, max_where_col_nums, T_q)
        where_op_one_hot_padded = self.get_where_op_one_hot_padded(where_op_idxes, where_nums, where_col_idxes, n_cond_ops=self.n_cond_ops)#.to(o_q.device)  # (B, max_where_col_nums, n_cond_ops)
        where_op = self.where_op_linear(where_op_one_hot_padded)  # (B, max_where_col_nums, H)
        
        vec = torch.cat([col_q_context, col_context_padded, where_op], dim=2)  # (B, max_where_col_nums, 3H)
        max_where_col_nums = vec.size(1)
        # predict each where_col
        total_scores = []
        for i in range(max_where_col_nums):
            g_wv_tkns_i = torch.LongTensor([g_wv_tkns[b_idx][i] for b_idx in range(batch_size)]) if g_wv_tkns is not None else None  # (B, T_d_i)
            vec_i = vec[:, i, :]  # (B, 3H)
            
            h_0 = self.output_lstm_hidden_init_linear(vec_i).unsqueeze(1).transpose(0, 1).contiguous()  # (B, 3H) -> (B, bert_H) -> (1, B, bert_H)
            c_0 = self.output_lstm_cell_init_linear(vec_i).unsqueeze(1).transpose(0, 1).contiguous()  # (B, 3H) -> (B, bert_H) -> (1, B, bert_H)
            
            scores = self.decode_single_where_col(batch_size, h_0, c_0, value_tkn_max_len=value_tkn_max_len, g_wv_tkns_i=g_wv_tkns_i)  # (B, T_d_i, vocab_size)
            total_scores.append(scores)
        
        # total_scores: [(B, T_d_i, vocab_size)] x max_where_col_nums
        return total_scores
    
    def start_token(self, batch_size: int):
        sos = torch.LongTensor([self.start_tkn_id]*batch_size).unsqueeze(1)  # (B, 1)
        return sos
    
    def decode_single_where_col(self, batch_size: int, h_0: torch.Tensor, c_0: torch.Tensor, value_tkn_max_len: Union[None, int]=None, g_wv_tkns_i: Union[None, List[List[int]]]=None):
        if value_tkn_max_len is None:
            # [Training] set the max length to gold token max length (already padded)
            max_len = len(g_wv_tkns_i[0])
        else:
            # [Testing]  don't know the max length
            max_len = value_tkn_max_len
            
        sos = self.start_token(batch_size)  # (B, 1)
        emb = self.embedding_layer(sos)  # (B, 1, bert_H)
        scores = [] 
        for i in range(max_len):
            o, (h, c) = self.output_lstm(emb, (h_0, c_0))  # h: (1, B, bert_H)  
            s = self.output_linear(h[-1, :]) # select last layer if use multiple rnn layers, h: (1, B, bert_H) -> (B, bert_H) -> s: (B, vocab_size)
            scores.append(s)
            
            if g_wv_tkns_i is not None:
                # [Training] Teacher Force model
                pred = g_wv_tkns_i[:, i]  # (B, )
            else:
                # [Testing]
                pred = s.argmax(1)  # (1,) only for single batch_size
                if pred.item() == self.end_tkn_id:
                    break
                    
            emb = self.embedding_layer(pred.unsqueeze(1))  # (B, 1, bert_H)
        
        return torch.stack(scores).transpose(0, 1).contiguous() # (T_d_i, B, vocab_size) -> (B, T_d_i, vocab_size)
        
    def get_context_padded(self, col_context: torch.Tensor, where_nums: List[int], where_col_idxes: List[List[int]]):
        r"""
        Select the where column index and pad if some batch doesn't match the max length of tensor
        In case for have different where column lengths
        """
        batch_size, n_col, hidden_size = col_context.size()
        max_where_col_nums = max(where_nums)
        batches = [col_context[i, batch_col] for i, batch_col in enumerate(where_col_idxes)]  # [(where_col_nums, hidden_size), ...]  len = B
        batches_padded = []
        for b in batches:
            where_col_nums = b.size(0)
            if where_col_nums < max_where_col_nums:
                b_padded = torch.cat([b, torch.zeros((max_where_col_nums-where_col_nums), hidden_size)], dim=0)
            else:
                b_padded = b
            batches_padded.append(b_padded)  # (max_where_col_nums, hidden_size)
            
        return torch.stack(batches_padded) # (B, max_where_col_nums, hidden_size)
    
    
    def get_where_op_one_hot_padded(self, where_op_idxes: List[List[int]], where_nums: List[int], where_col_idxes: List[List[int]], n_cond_ops: int):
        r"""
        Turn where operation indexs into one hot encoded vectors
        In case for have different where column lengths
        """
        max_where_col_nums = max(where_nums)
        batches = [torch.zeros(where_num, n_cond_ops).scatter(1, torch.LongTensor(batch_col).unsqueeze(1), 1) for where_num, batch_col in zip(where_nums, where_op_idxes)]  
        # batches = [(where_col_nums, n_cond_ops), ...]  len = B
        batches_padded = []
        for b in batches:
            where_col_nums = b.size(0)
            if where_col_nums < max_where_col_nums:
                b_padded = torch.cat([b, torch.zeros((max_where_col_nums-where_col_nums), n_cond_ops)], dim=0)
            else:
                b_padded = b
            batches_padded.append(b_padded)  # (max_where_col_nums, hidden_size)
        return torch.stack(batches_padded) # (B, max_where_col_nums, hidden_size)