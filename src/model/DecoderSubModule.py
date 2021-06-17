import torch
import torch.nn as nn
from .AttentionModule import C2QAttention, SelfAttention
from typing import List, Dict, Any, Tuple, Union
from copy import deepcopy

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

    def forward(self, question_padded, db_padded, col_padded, question_lengths: List[int], col_lengths: List[int], rt_attn=False):
        r"""
        predict column index
        """
        batch_size, n_col, _ = col_padded.size()
        o_q, _ = self.lstm_q(question_padded)  # o_q: (B, T_q, H)
        o_c, _ = self.lstm_h(col_padded)  # o_c: (B, T_c, H)
        _, (h_h, _) = self.lstm_h(db_padded)  # h_h: (n_direc*num_layers, B, H/2)
        
        header_summary = torch.cat([h for h in h_h[-2:]], dim=1).unsqueeze(1).expand(batch_size, n_col, -1)  # (B, T_c, H)
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
                
    def forward(self, question_padded, db_padded, col_padded, question_lengths: List[int], col_lengths: List[int], select_idxes: List[int], rt_attn=False):
        r"""
        predict agg index
        select_prob: selected argmax indices of select_output score
        """
        batch_size, n_col, _ = col_padded.size()
        o_q, _ = self.lstm_q(question_padded)  # o_q: (B, T_q, H)
        o_c, _ = self.lstm_h(col_padded)  # o_c: (B, T_c, H)
        _, (h_h, _) = self.lstm_h(db_padded)  # h_h: (n_direc*num_layers, B, H/2)
        
        header_summary = torch.cat([h for h in h_h[-2:]], dim=1).unsqueeze(1).expand(batch_size, n_col, -1)  # (B, T_c, H)
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
        
    def forward(self, question_padded, db_padded, col_padded, question_lengths: List[int], col_lengths: List[int], rt_attn=False):
        r"""
        predict agg index
        select_prob: selected argmax indices of select_output score
        """
        batch_size, n_col, _ = col_padded.size()
        o_c, _ = self.lstm_h(col_padded)  # o_c: (B, T_c, H)
        _, (h_h, _) = self.lstm_h(db_padded)  # h_h: (n_direc*num_layers, B, H/2)
        
        header_summary = torch.cat([h for h in h_h[-2:]], dim=1).unsqueeze(1).expand(batch_size, n_col, -1)  # (B, T_c, H)
        col_context = torch.cat([o_c, header_summary], dim=2)  # (B, T_c, 2H)

        col_self_attn, col_attn = self.col_self_attn(col_context, col_lengths, rt_attn)  # (B, 2H), (B, T_c)

        h_0 = self.lstm_q_hidden_init_linear(col_self_attn)  # (B, 2H)
        h_0 = h_0.view(batch_size, 2*self.num_layers, -1).transpose(0, 1).contiguous()  # (B, n_direc*num_layers, H/2) -> (n_direc*num_layers, B, H/2)
        c_0 = self.lstm_q_cell_init_linear(col_self_attn)  # (B, 2H)
        c_0 = c_0.view(batch_size, 2*self.num_layers, -1).transpose(0, 1).contiguous()  # (B, n_direc*num_layers, H/2) -> (n_direc*num_layers, B, H/2)
        
        o_q, _ = self.lstm_q(question_padded, (h_0, c_0))  # o_q: (B, T_q, H)
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

    def forward(self, question_padded, db_padded, col_padded, question_lengths: List[int], col_lengths: List[int], rt_attn=False):
        r"""
        predict column index
        """
        batch_size, n_col, _ = col_padded.size()
        o_q, _ = self.lstm_q(question_padded)  # o_q: (B, T_q, H)
        o_c, _ = self.lstm_h(col_padded)  # o_c: (B, T_c, H)
        _, (h_h, _) = self.lstm_h(db_padded)  # h_h: (n_direc*num_layers, B, H/2)
        
        header_summary = torch.cat([h for h in h_h[-2:]], dim=1).unsqueeze(1).expand(batch_size, n_col, -1)  # (B, T_c, H)
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
    
    def forward(self, question_padded, db_padded, col_padded, question_lengths: List[int], where_nums: List[int], where_col_idxes: List[List[int]], rt_attn=False):
        r"""
        predict agg index
        select_prob: selected argmax indices of select_output score
        max_where_col_nums is settled at WhereColumnDecoder, but it can be lower than or equal to `max_where_conds`
        """
        device = col_padded.device
        batch_size, n_col, _ = col_padded.size()
        o_q, _ = self.lstm_q(question_padded)  # o_q: (B, T_q, H)
        o_c, _ = self.lstm_h(col_padded)  # o_c: (B, T_c, H)
        _, (h_h, _) = self.lstm_h(db_padded)  # h_h: (n_direc*num_layers, B, H/2)
        
        header_summary = torch.cat([h for h in h_h[-2:]], dim=1).unsqueeze(1).expand(batch_size, n_col, -1)  # (B, T_c, H)
        col_context = torch.cat([o_c, header_summary], dim=2)  # (B, T_c, 2H)
        col_context = self.col_context_linear(col_context)  # (B, T_c, H)
        # (B, max_where_col_nums, H), (B,)
        col_context_padded = self.get_context_padded(col_context, where_col_idxes, device)
        
        # (B, max_where_col_nums, H), (B, max_where_col_nums, T_q)
        col_q_context, attn = self.col2question_attn(col_context_padded, o_q, question_lengths, where_nums, rt_attn)  
        vec = torch.cat([col_q_context, col_context_padded], dim=2)  # (B, max_where_col_nums, 2H)
        output = self.output_layer(vec)  # (B, max_where_col_nums, n_cond_ops)
        # TODO: add penalty for padded header(column) information
#         for i, l in enumerate(where_nums):
#             output[:, i, :] = -1e10
        return output, attn
        
    def get_context_padded(self, col_context, where_col_idxes, device: str="cpu"):
        r"""
        Select the where column index and pad if some batch doesn't match the max length of tensor
        In case for have different where column lengths
        """
        hidden_size = col_context.size(2)
        max_where_col_nums = self.max_where_conds # max(where_nums) 
        batches = [col_context[i, batch_col] for i, batch_col in enumerate(where_col_idxes)]  # [(where_col_nums, hidden_size), ...]  len = B
        batches_padded = []
        for b in batches:
            where_col_nums = b.size(0)
            if where_col_nums < max_where_col_nums:
                
#                 self.register_buffer("pad_zeros_context", torch.zeros((max_where_col_nums-where_col_nums), hidden_size, device=device))
#                 b_padded = torch.cat([b, self.pad_zeros_context], dim=0)
                # Use Register Buffer to code following code for PyTroch Lightning
                b_padded = torch.cat([b, torch.zeros((max_where_col_nums-where_col_nums), hidden_size, device=device).contiguous()], dim=0)
            else:
                b_padded = b
            batches_padded.append(b_padded)  # (max_where_col_nums, hidden_size)
        return torch.stack(batches_padded) # (B, max_where_col_nums, hidden_size), (B,)
    
    
class WhereValueDecoder(nn.Module):
    r"""WHERE Value Decoder"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int=2, dropout_ratio: float=0.3, max_where_conds: int=4, n_cond_ops: int=4,
                 start_tkn_id=8002, end_tkn_id=8003, embedding_layer=None) -> None:
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
            if not isinstance(embedding_layer, torch.nn.modules.sparse.Embedding):
                embedding_layer = embedding_layer.word_embeddings
            self.embedding_layer = embedding_layer
            vocab_size, bert_hidden_size = embedding_layer.weight.data.size()
            self.output_lstm_hidden_init_linear = nn.Linear(3*hidden_size, bert_hidden_size)
            self.output_lstm_cell_init_linear = nn.Linear(3*hidden_size, bert_hidden_size)
            self.output_lstm = nn.LSTM(bert_hidden_size, bert_hidden_size, 1, batch_first=True)
            self.output_linear = nn.Linear(bert_hidden_size, vocab_size)
            self.output_linear.weight.data = embedding_layer.weight.data
        
    def forward(self, question_padded, db_padded, col_padded, question_lengths: List[int], where_nums: List[int], where_col_idxes: List[List[int]], where_op_idxes: List[List[int]], value_tkn_max_len=None, g_wv_tkns=None, rt_attn=False):
        r"""
        predict agg index
        select_prob: selected argmax indices of select_output score
        max_where_col_nums is setted at WhereColumnDecoder
        value_tkn_max_len = Train if None else Test
        g_wv_tkns = When Train should not be None
        
        """
        
        device = col_padded.device
        batch_size, n_col, _ = col_padded.size()
        o_q, _ = self.lstm_q(question_padded)  # o_q: (B, T_q, H)
        o_c, _ = self.lstm_h(col_padded)  # o_c: (B, T_c, H)
        _, (h_h, _) = self.lstm_h(db_padded)  # h_h: (n_direc*num_layers, B, H/2)
        
        header_summary = torch.cat([h for h in h_h[-2:]], dim=1).unsqueeze(1).expand(batch_size, n_col, -1)  # (B, T_c, H)
        col_context = torch.cat([o_c, header_summary], dim=2)  # (B, T_c, 2H)
        col_context = self.col_context_linear(col_context)  # (B, T_c, H)
        col_context_padded = self.get_context_padded(col_context, where_col_idxes, device)  # (B, max_where_col_nums, H)

        col_q_context, attn = self.col2question_attn(col_context_padded, o_q, question_lengths, where_nums, rt_attn)  # (B, max_where_col_nums, H), (B, max_where_col_nums, T_q)
        where_op_one_hot_padded = self.get_where_op_one_hot_padded(
            where_op_idxes, where_nums, where_col_idxes, n_cond_ops=self.n_cond_ops, device=device)  # (B, max_where_col_nums, n_cond_ops)

        where_op = self.where_op_linear(where_op_one_hot_padded)  # (B, max_where_col_nums, H)

        vec = torch.cat([col_q_context, col_context_padded, where_op], dim=2)  # (B, max_where_col_nums, 3H)
        
        # predict each where_col
        total_scores = []
        # max_where_col_nums = vec.size(1)
        # for i in range(max_where_col_nums):
        if g_wv_tkns is not None:
            g_wv_tkns = deepcopy(g_wv_tkns)
            g_max_len_where_num = len(list(zip(*g_wv_tkns)))
            if g_max_len_where_num < self.max_where_conds:
                for b in range(batch_size):
                    g_wv_tkns[b].extend([[self.end_tkn_id]]*(self.max_where_conds-g_max_len_where_num))

        for i in range(self.max_where_conds):
            g_wv_tkns_i = torch.LongTensor([g_wv_tkns[b_idx][i] for b_idx in range(batch_size)]).to(device).contiguous() if g_wv_tkns is not None else None  # (B, T_d_i)
            vec_i = vec[:, i, :]  # (B, 3H)
            
            h_0 = self.output_lstm_hidden_init_linear(vec_i).unsqueeze(1).transpose(0, 1).contiguous()  # (B, 3H) -> (B, bert_H) -> (1, B, bert_H)
            c_0 = self.output_lstm_cell_init_linear(vec_i).unsqueeze(1).transpose(0, 1).contiguous()  # (B, 3H) -> (B, bert_H) -> (1, B, bert_H)
            
            scores = self.decode_single_where_col(batch_size, h_0, c_0, value_tkn_max_len=value_tkn_max_len, g_wv_tkns_i=g_wv_tkns_i, device=device)  # (B, T_d_i, vocab_size)
            total_scores.append(scores)
        
        # total_scores: [(B, T_d_i, vocab_size)] x max_where_col_nums
        return total_scores, attn
    
    def start_token(self, batch_size, device):
#         self.register_buffer("sos", torch.LongTensor([self.start_tkn_id]*batch_size).to(device).unsqueeze(1))
        sos = torch.LongTensor([self.start_tkn_id]*batch_size).unsqueeze(1).to(device).contiguous()  # (B, 1)
        return sos
    
    def decode_single_where_col(self, batch_size, h_0, c_0, value_tkn_max_len=None, g_wv_tkns_i=None, device: str="cpu"):
        if value_tkn_max_len is None:
            # [Training] set the max length to gold token max length (already padded)
            max_len = len(g_wv_tkns_i[0])
        else:
            # [Testing]  don't know the max length
            max_len = value_tkn_max_len
            
        # Version2: left_batch_size = batch_size
        
        sos = self.start_token(batch_size, device)  # (B, 1)
        emb = self.embedding_layer(sos)  # (B, 1, bert_H)
        scores = [] 
        for i in range(max_len):
            _, (h, _) = self.output_lstm(emb, (h_0, c_0))  # h: (1, B, bert_H)  
            s = self.output_linear(h[-1, :]) # select last layer if use multiple rnn layers, h: (1, B, bert_H) -> (B, bert_H) -> s: (B, vocab_size)
            scores.append(s)
            if g_wv_tkns_i is not None:
                # [Training] Teacher Force model
                pred = g_wv_tkns_i[:, i]  # (B, )
            else:
                # [Testing]
                pred = s.argmax(1)  # (B, )
                if (pred == self.end_tkn_id).sum() == batch_size:  # all stop
                    break
                # Version2: Seperate all tokens
                # if (pred == dd.end_tkn_id).sum() == left_batch_size:  # all stop
                #     scores.append(s)
                #     break
                # else:
                #     stop_mask = pred == dd.end_tkn_id
                #     pred = pred[~stop_mask]
                #     scores.append(pred)
                #     left_batch_size -= stop_mask.sum().item()
                    
            emb = self.embedding_layer(pred.unsqueeze(1))  # (B, 1, bert_H)
            
        return torch.stack(scores).transpose(0, 1).contiguous() # (T_d_i, B, vocab_size) -> (B, T_d_i, vocab_size)
        
    def get_context_padded(self, col_context: torch.Tensor, where_col_idxes: List[List[int]], device: str="cpu"):
        r"""
        Select the where column index and pad if some batch doesn't match the max length of tensor
        In case for have different where column lengths
        """
        hidden_size = col_context.size(2)
        max_where_col_nums = self.max_where_conds # max(where_nums)
        batches = [col_context[i, batch_col] for i, batch_col in enumerate(where_col_idxes)]  # [(where_col_nums, hidden_size), ...]  len = B
        batches_padded = []
        for b in batches:
            where_col_nums = b.size(0)
            if where_col_nums < max_where_col_nums:

#                 self.register_buffer("pad_zeros_context", torch.zeros((max_where_col_nums-where_col_nums), hidden_size, device=device))
#                 b_padded = torch.cat([b, self.pad_zeros_context], dim=0)
                b_padded = torch.cat([b, torch.zeros((max_where_col_nums-where_col_nums), hidden_size, device=device).contiguous()], dim=0)
            else:
                b_padded = b
            batches_padded.append(b_padded)  # (max_where_col_nums, hidden_size)
        return torch.stack(batches_padded) # (B, max_where_col_nums, hidden_size)
    
    def get_where_op_one_hot_padded(self, where_op_idxes: List[List[int]], where_nums: List[int], where_col_idxes: List[List[int]], n_cond_ops: int, device: str="cpu"):
        r"""
        Turn where operation indexs into one hot encoded vectors
        In case for have different where column lengths
        """
        max_where_col_nums = self.max_where_conds # max(where_nums)
        batches = [
            torch.zeros(where_num, n_cond_ops).scatter(1, torch.LongTensor(batch_col).unsqueeze(1), 1).to(device).contiguous() 
            for where_num, batch_col in zip(where_nums, where_op_idxes)
        ]  
        # batches = [(where_col_nums, n_cond_ops), ...]  len = B
        batches_padded = []
        for b in batches:
            where_col_nums = b.size(0)
            if where_col_nums < max_where_col_nums:
#                 self.register_buffer("pad_zeros_where_op", torch.zeros((max_where_col_nums-where_col_nums), n_cond_ops, device=device))
#                 b_padded = torch.cat([b, self.pad_zeros_where_op], dim=0)
                b_padded = torch.cat([b, torch.zeros((max_where_col_nums-where_col_nums), n_cond_ops, device=device).contiguous()], dim=0)
            else:
                b_padded = b
            batches_padded.append(b_padded)  # (max_where_col_nums, hidden_size)
            
        return torch.stack(batches_padded) # (B, max_where_col_nums, hidden_size)