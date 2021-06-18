import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Union
from .DecoderSubModule import SelectDecoder, AggDecoder, WhereNumDecoder, WhereColumnDecoder, WhereOpDecoder, WhereValueDecoder

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_ratio, max_where_conds, n_agg_ops, n_cond_ops, start_tkn_id, end_tkn_id, embedding_layer):
        super().__init__()
        self.max_where_conds = max_where_conds
        
        self.select_decoder = SelectDecoder(
            input_size, hidden_size, output_size=1, num_layers=num_layers, dropout_ratio=dropout_ratio
        )
        self.agg_decoder = AggDecoder(
            input_size, hidden_size, output_size=n_agg_ops, num_layers=num_layers, dropout_ratio=dropout_ratio
        )
        self.where_num_decoder = WhereNumDecoder(
            input_size, hidden_size, output_size=(max_where_conds+1), num_layers=num_layers, dropout_ratio=dropout_ratio
        )
        self.where_col_decoder = WhereColumnDecoder(
            input_size, hidden_size, output_size=1, num_layers=num_layers, dropout_ratio=dropout_ratio, max_where_conds=max_where_conds
        )
        self.where_op_decoder = WhereOpDecoder(
            input_size, hidden_size, output_size=n_cond_ops, num_layers=num_layers, dropout_ratio=dropout_ratio, max_where_conds=max_where_conds
        )
        self.where_value_decoder = WhereValueDecoder(
            input_size, hidden_size, output_size=n_cond_ops, num_layers=num_layers, dropout_ratio=dropout_ratio, max_where_conds=max_where_conds, 
            n_cond_ops=n_cond_ops, start_tkn_id=start_tkn_id, end_tkn_id=end_tkn_id, embedding_layer=embedding_layer
        )
    
    
    def forward(self, question_padded, db_padded, col_padded, question_lengths, col_lengths, value_tkn_max_len=None, gold=None, rt_attn=False):
        """
        # Outputs Size
        # sc = (B, T_c)
        # sa = (B, n_agg_ops)
        # wn = (B, max_where_conds+1)
        # wc = (B, T_c): binary
        # wo = (B, max_where_col_nums, n_cond_ops)
        # wv = [(B, T_d_i, vocab_size)] x max_where_col_nums / T_d_i = may have different length for answer
        """
        if gold is None:
            g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_tkns = [None] * 6
        else:
            g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_tkns = gold

        select_outputs, attn_c2q_sc = self.select_decoder(question_padded, db_padded, col_padded, question_lengths, col_lengths, rt_attn)
        select_idxes = g_sc if g_sc else self.predict_decoder("sc", select_outputs=select_outputs)

        agg_outputs, attn_c2q_sa = self.agg_decoder(question_padded, db_padded, col_padded, question_lengths, col_lengths, select_idxes, rt_attn)

        where_num_outputs, (attn_self_col_wn, attn_self_cntxt_wn)  = self.where_num_decoder(question_padded, db_padded, col_padded, question_lengths, col_lengths, rt_attn)
        where_nums = g_wn if g_wn else self.predict_decoder("wn", where_num_outputs=where_num_outputs)

        where_col_outputs, attn_c2q_wc = self.where_col_decoder(question_padded, db_padded, col_padded, question_lengths, col_lengths, rt_attn)
        where_col_idxes = g_wc if g_wc else self.predict_decoder("wc", where_col_outputs=where_col_outputs, where_nums=where_nums)

        where_op_outputs, attn_c2q_wo = self.where_op_decoder(question_padded, db_padded, col_padded, question_lengths, where_nums, where_col_idxes, rt_attn)
        where_op_idxes = g_wo if g_wo else self.predict_decoder("wo", where_op_outputs=where_op_outputs, where_nums=where_nums)

        where_value_outputs, attn_c2q_wv = self.where_value_decoder(
            question_padded, db_padded, col_padded, question_lengths, where_nums, where_col_idxes, where_op_idxes, value_tkn_max_len, g_wv_tkns, rt_attn
        )
        
        decoder_outputs = {
            "sc": select_outputs,  # cross entropy
            "sa": agg_outputs,  # cross entropy
            "wn": where_num_outputs,  # cross entropy
            "wc": where_col_outputs,  # binary cross entropy
            "wo": where_op_outputs,  # cross entropy
            "wv": where_value_outputs  # cross entropy
        }
        
        decoder_attns = {
            "sc": attn_c2q_sc,  # cross entropy
            "sa": attn_c2q_sa,  # cross entropy
            "wn": (attn_self_col_wn, attn_self_cntxt_wn),  # cross entropy
            "wc": attn_c2q_wc,  # binary cross entropy
            "wo": attn_c2q_wo,  # cross entropy
            "wv": attn_c2q_wv  # cross entropy
        }
        
        return decoder_outputs, decoder_attns
        
    def predict_decoder(self, typ, **kwargs):
        r"""
        if not using teacher force model will use this function to predict answer
        # Outputs Size
        # sc = (B, T_c)
        # sa = (B, n_agg_ops)
        # wn = (B, max_where_conds+1)
        # wc = (B, T_c): binary
        # wo = (B, max_where_col_nums, n_cond_ops)
        # wv = [(B, T_d_i, vocab_size)] x max_where_col_nums / T_d_i = may have different length for answer
        """
        if typ == "sc":  # SELECT column
            select_outputs = kwargs["select_outputs"]
            return select_outputs.argmax(1).tolist()
        elif typ == "sa":  # SELECT aggregation operator
            agg_outputs = kwargs["agg_outputs"]
            return agg_outputs.argmax(1).tolist()
        elif typ == "wn":  # WHERE number
            where_num_outputs = kwargs["where_num_outputs"]
            return where_num_outputs.argmax(1).tolist()
        elif typ == "wc":  # WHERE clause column
            where_col_outputs = kwargs["where_col_outputs"]
            where_col_argsort = torch.sigmoid(where_col_outputs).argsort(1)
            where_nums = kwargs["where_nums"]
            where_col_idxes = [where_col_argsort[b_idx, :w_num].tolist() for b_idx, w_num in enumerate(where_nums)]
            return where_col_idxes
        elif typ == "wo":  # WHERE clause operator
            where_op_outputs = kwargs["where_op_outputs"]
            where_nums = kwargs["where_nums"]
            where_op_idxes = []
            for b_idx, w_num in enumerate(where_nums):
                if w_num == 0:  # means no where number
                    where_op_idxes.append([])
                else:
                    where_op_idxes.append(where_op_outputs.argmax(2)[b_idx, :w_num].tolist())
            return where_op_idxes
        elif typ == "wv":  # WHERE clause value
            where_value_outputs = kwargs["where_value_outputs"]
            return [o.argmax(2).tolist() for o in where_value_outputs]  # iter with each where clause
        else:
            raise KeyError("`typ` must be in ['sc', 'sa', 'wn', 'wc', 'wo', 'wv']")