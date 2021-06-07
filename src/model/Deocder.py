import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Union

class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout_ratio: float, max_where_conds: int, n_agg_ops: int, n_cond_ops: int, start_tkn_id: int, end_tkn_id: int, value_tkn_max_len: int, embedding_layer: torch.nn.modules.sparse.Embedding) -> None:
        super().__init__()
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
    
    
    def forward(self, question_padded: torch.Tensor, header_padded: torch.Tensor, col_padded: torch.Tensor, question_lengths: torch.Tensor, col_lengths: torch.Tensor, gold: List[List[Any]]):
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_tkns = gold
        decoder_outputs = {}

        select_outputs, _ = self.select_decoder(question_padded, header_padded, col_padded, question_lengths, col_lengths)
        select_idxes = g_sc if g_sc else self.predict_decoder("sc", select_outputs=select_outputs)

        agg_outputs, _ = self.agg_decoder(question_padded, col_padded, question_lengths, col_lengths, select_idxes)

        where_num_outputs, _  = self.where_num_decoder(question_padded, header_padded, col_padded, question_lengths, col_lengths)
        where_nums = g_wn if g_wn else self.predict_decoder("wn", where_num_outputs=where_num_outputs)

        where_col_outputs, _ = self.where_col_decoder(question_padded, header_padded, col_padded, question_lengths, col_lengths)
        where_col_argsort = torch.sigmoid(where_col_outputs).argsort(1)
        where_col_idxes = g_wc if g_wc else self.predict_decoder("wc", where_col_argsort=where_col_argsort, where_nums=where_nums)

        where_op_outputs = self.where_op_decoder(question_padded, col_padded, question_lengths, where_nums, where_col_idxes)
        where_op_idxes = g_wo if g_wo else self.predict_decoder("wo", where_op_outputs=where_op_outputs, where_nums=where_nums)

        where_value_outputs = self.where_value_decoder(question_padded, col_padded, question_lengths, where_nums, where_col_idxes, where_op_idxes, value_tkn_max_len, g_wv_tkns)

        decoder_outputs = {
            "sc": select_outputs,
            "sa": agg_outputs,
            "wn": where_num_outputs,
            "wc": where_col_outputs,
            "wo": where_op_outputs,
            "wv": where_value_outputs
        }
        
        return decoder_outputs
        
    def predict_decoder(typ, **kwargs):
        r"""
        if not using teacher force model will use this function to predict answer
        """
        if typ == "sc":  # SELECT column
            select_outputs = kwargs["select_outputs"]
            return select_outputs.argmax(1).tolist()
        elif typ == "sa":  # SELECT aggregation operator
            # not need actually
            agg_outputs = kwargs["agg_outputs"]
            return agg_outputs.argmax(1)
        elif typ == "wn":  # WHERE number
            where_num_outputs = kwargs["where_num_outputs"]
            return where_num_outputs.argmax(1).tolist()
        elif typ == "wc":  # WHERE clause column
            where_col_argsort = kwargs["where_col_argsort"]
            where_nums = kwargs["where_nums"]
            where_col_idxes = [where_col_argsort[b_idx, :w_num].tolist() for b_idx, w_num in enumerate(where_nums)]
            return where_col_idxes
        elif typ == "wo":  # WHERE clause operator
            where_op_outputs = kwargs["where_op_outputs"]
            where_nums = kwargs["where_nums"]
            where_op_idxes = [where_op_outputs.argmax(2)[b_idx, :w_num].tolist() for b_idx, w_num in enumerate(where_nums)]
            return where_op_idxes
        elif typ == "wv":  # WHERE clause value
            # not need actually
            where_value_outputs = kwargs["where_value_outputs"]
            return [o.argmax(2) for o in where_value_outputs]
        else:
            raise KeyError("`typ` must be in ['sc', 'sa', 'wn', 'wc', 'wo', 'wv']")