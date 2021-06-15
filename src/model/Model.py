__author__ = "simonjisu"
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim

import transformers
import torchmetrics
import pytorch_lightning as pl

from pathlib import Path
from transformers import BertModel, BertConfig
from typing import Tuple, Dict, List, Union, Any

from .DeocderModule import Decoder
from .KoBertTokenizer import KoBertTokenizer
from .dbengine import DBEngine
from .utils import Perplexity

class Text2SQL(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dbengine = DBEngine(Path(self.hparams.db_path))
        self.n_agg_ops = len(self.dbengine.agg_ops)
        self.n_cond_ops = len(self.dbengine.cond_ops)
        # Encoder
        self.model_bert, self.tokenizer_bert, self.config_bert = self.get_bert(model_path=self.hparams.model_bert_path)
        # Decoder
        self.model_decoder = Decoder(
            input_size=self.config_bert.hidden_size, 
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            dropout_ratio=self.hparams.dropout_ratio,
            max_where_conds=self.hparams.max_where_conds,
            n_agg_ops=self.n_agg_ops,
            n_cond_ops=self.n_cond_ops,
            start_tkn_id = self.tokenizer_bert.additional_special_tokens_ids[0],
            end_tkn_id = self.tokenizer_bert.additional_special_tokens_ids[1],
            embedding_layer = self.model_bert.embeddings.word_embeddings
        )
        
        # Loss function & Metrics
        self.vocab_size = len(self.tokenizer_bert)
        self.totensor = lambda x: torch.LongTensor(x).to(self.device)
        self.table = None
        self.create_metrics()

    def get_bert(self, model_path: str, output_hidden_states: bool=False):
        self.special_tokens = [self.hparams.special_start_tkn, self.hparams.special_end_tkn, self.hparams.special_col_tkn] # sequence start, sequence end, column tokens
        tokenizer = KoBertTokenizer.from_pretrained(model_path, add_special_tokens=True, additional_special_tokens=self.special_tokens)
        config = BertConfig.from_pretrained(model_path)
        config.output_hidden_states = output_hidden_states

        model = BertModel.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
        model.config.output_hidden_states = output_hidden_states

        return model, tokenizer, config
    
    def create_metrics(self):
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")
        self.binary_cross_entropy = nn.BCEWithLogitsLoss(reduction="sum")
        
        self.acc_sc = torchmetrics.Accuracy()
        self.acc_sa = torchmetrics.Accuracy(num_classes=self.n_agg_ops)
        self.acc_wn = torchmetrics.Accuracy(num_classes=self.hparams.max_where_conds+1)
        self.acc_wo = torchmetrics.Accuracy(num_classes=self.n_cond_ops+1) # add one to calculate if where number is missing
        self.pp_wv = Perplexity()

    def reset_metrics_epoch_end(self):
        self.acc_sc.reset()
        self.acc_sa.reset()
        self.acc_wn.reset()
        self.acc_wo.reset()
        self.pp_wv.reset()
        
    def forward(self, batch_qs, batch_ts, batch_sqls=None, value_tkn_max_len=None, train=True):
        outputs = self.forward_outputs(batch_qs, batch_ts, batch_sqls, value_tkn_max_len, train)
        g_sc, g_sa, g_wn, g_wc, g_wo, _, g_wv_tkns = self.get_sql_answers(batch_sqls)
        gold = [g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_tkns]
        loss = self.calculate_loss(outputs, gold)  # when calculate loss must need gold answer
        return loss, outputs
            
    def forward_outputs(self, batch_qs, batch_ts, batch_sqls=None, value_tkn_max_len=None, train=True):
        # --- Get Answer & Variables ---
        if train:
            assert value_tkn_max_len is None, "In train phase, `value_tkn_max_len` must be None"
            assert batch_sqls is not None, "In train phase, `batch_sqls` must not be None"
            g_sc, g_sa, g_wn, g_wc, g_wo, _, g_wv_tkns = self.get_sql_answers(batch_sqls)
            gold = [g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_tkns]
        else:
            assert value_tkn_max_len is not None, "In validation Phase, `value_tkn_max_len` must not be None"
            gold = None
            value_tkn_max_len = value_tkn_max_len
            
        # --- Get Inputs for Encoder --- 
        encode_inputs = self.tokenizer_bert(
            batch_qs, batch_ts, 
            max_length=512, padding=True, truncation=True, return_tensors="pt", 
            return_attention_mask=True, 
            return_special_tokens_mask=False, 
        ).to(self.device)  # encode_input doesn't return the cuda device
        
        # --- Forward Encoder ---
        encode_outputs = self.model_bert(**encode_inputs)
        
        # --- Get Inputs for Decoder ---
        input_question_mask, input_table_mask, input_header_mask, input_col_mask = self.get_input_mask_and_answer(encode_inputs, self.tokenizer_bert)
        question_padded, question_lengths = self.get_decoder_batches(encode_outputs, input_question_mask, pad_idx=self.tokenizer_bert.pad_token_id)
        # table_padded, table_lengths = self.get_decoder_batches(encode_outputs, input_table_mask, pad_idx=self.tokenizer_bert.pad_token_id)  # Not used yet
        header_padded, header_lengths = self.get_decoder_batches(encode_outputs, input_header_mask, pad_idx=self.tokenizer_bert.pad_token_id)
        col_padded, col_lengths = self.get_decoder_batches(encode_outputs, input_col_mask, pad_idx=self.tokenizer_bert.pad_token_id)
        
        # --- Forward Decoder ---
        decoder_outputs = self.model_decoder(
            question_padded, 
            header_padded, 
            col_padded, 
            question_lengths, 
            col_lengths, 
            value_tkn_max_len, 
            gold
        )
        
        return decoder_outputs

    def get_input_mask_and_answer(self, encode_input: transformers.tokenization_utils_base.BatchEncoding, tokenizer: KoBertTokenizer) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """[summary]

        In this code 'table' means database table name(id), 'header' means database header, 'col' means index of header 

        Args:
            encode_input (transformers.tokenization_utils_base.BatchEncoding): [description]
            tokenizer (KoBertTokenizer): [description]

        Returns:
            Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]: [description]
        """

        batch_size, max_length = encode_input["input_ids"].size()
        sep_tkn_mask = encode_input["input_ids"] == tokenizer.sep_token_id
        start_tkn_id, end_tkn_id, col_tkn_id = tokenizer.additional_special_tokens_ids

        input_question_mask = torch.bitwise_and(encode_input["token_type_ids"] == 0, encode_input["attention_mask"].bool())
        input_question_mask = torch.bitwise_and(input_question_mask, ~sep_tkn_mask) # [SEP] mask out
        input_question_mask[:, 0] = False  # [CLS] mask out

        db_mask = torch.bitwise_and(encode_input["token_type_ids"] == 1, encode_input["attention_mask"].bool())
        db_mask = torch.bitwise_xor(db_mask, sep_tkn_mask)
        col_tkn_mask = encode_input["input_ids"] == col_tkn_id
        db_mask = torch.bitwise_and(db_mask, ~col_tkn_mask)
        # split table_mask and header_mask
        input_idx = torch.arange(max_length).repeat(batch_size, 1).to(self.device)
        db_idx = input_idx[db_mask]
        table_header_tkn_idx = db_idx[db_idx > 0]
        table_start_idx = table_header_tkn_idx.view(batch_size, -1)[:, 0] + 1
        start_idx = table_header_tkn_idx[1:][table_header_tkn_idx.diff() == 2].view(batch_size, -1)
        table_end_sep_idx = start_idx[:, 0] - 1
        split_size = torch.stack([
            table_end_sep_idx-table_start_idx+1, table_header_tkn_idx.view(batch_size, -1).size(1)-(table_end_sep_idx-table_start_idx+1)
        ]).transpose(0, 1)

        # Token idx
        table_tkn_idx, header_tkn_idx = map(
            lambda x: torch.stack(x).to(self.device), 
            zip(*[torch.split(x, size.tolist()) for x, size in zip(table_header_tkn_idx.view(batch_size, -1), split_size)])
        )

        table_tkn_idx = table_tkn_idx[:, 1:]

        # TODO: [EXP] Experiment for generate column directly
        # If [EXP], `table_tkn_mask` and `header_tkn_mask` should include [S] & [E] tokens
        table_tkn_mask = torch.zeros_like(encode_input["input_ids"], dtype=torch.bool, device=self.device).scatter(1, table_tkn_idx, True)
        header_tkn_mask = torch.zeros_like(encode_input["input_ids"], dtype=torch.bool, device=self.device).scatter(1, header_tkn_idx, True)

        # TODO: [EXP] Experiment for generate column directly
        # For Decoder Input, Maskout [S], [E] for table & header -> will be done automatically
        input_table_mask = self.get_decoder_input_mask(
            encode_input["input_ids"], table_tkn_mask, batch_size, start_tkn_id, end_tkn_id
        )
        input_header_mask = self.get_decoder_input_mask(
            encode_input["input_ids"], header_tkn_mask, batch_size, start_tkn_id, end_tkn_id
        )

        # [COL] token mask: this is for attention
        col_tkn_idx = input_idx[col_tkn_mask].view(batch_size, -1)
        input_col_mask = torch.zeros_like(encode_input["input_ids"], device=self.device, dtype=torch.bool).scatter(1, col_tkn_idx, True)

        # TODO: [EXP] Experiment for generate column directly
        # For Answer, Maskout [S] for table & header 
        # answer_table_tkns = get_answer(
        #     encode_input["input_ids"], table_tkn_mask, batch_size, start_tkn_id, end_tkn_id
        # )
        # answer_header_tkns = get_answer(
        #     encode_input["input_ids"], header_tkn_mask, batch_size, start_tkn_id, end_tkn_id
        # )

        return input_question_mask, input_table_mask, input_header_mask, input_col_mask # , answer_table_tkns, answer_header_tkns    

    ## Masks
    # TODO: [EXP] Experiment for generate column directly
    # def get_answer(input_ids, mask, batch_size, start_tkn_id, end_tkn_id):
    #     r"""
    #     answer should include end token: [E]
    #     """
    #     masked_input_ids = input_ids[mask]
    #     start_tkn_mask = masked_input_ids == start_tkn_id
    #     end_tkn_mask = masked_input_ids == end_tkn_id
    #     table_col_length = masked_input_ids.view(batch_size, -1).size(1)
    #     start_end_mask = torch.bitwise_or(start_tkn_mask, end_tkn_mask)
    #     index = torch.arange(table_col_length).repeat(batch_size)[start_end_mask].view(batch_size, -1, 2)
    #     tkn_lengths = index[:, :, 1] - index[:, :, 0]
    #     answer_col_tkns = [x.split(tkn_length.tolist()) for x, tkn_length in zip(
    #         masked_input_ids[~start_tkn_mask].view(batch_size, -1), tkn_lengths)]
    #     return answer_col_tkns

    def get_decoder_input_mask(self, input_ids: torch.Tensor, mask: torch.BoolTensor, batch_size: int, start_tkn_id: int, end_tkn_id: int) -> torch.BoolTensor:
        """[summary]

        Args:
            input_ids (torch.Tensor): [description]
            mask (torch.BoolTensor): [description]
            batch_size (int): [description]
            start_tkn_id (int): [description]
            end_tkn_id (int): [description]

        Returns:
            torch.BoolTensor: [description]
        """    
        start_tkn_mask = input_ids == start_tkn_id
        end_tkn_mask = input_ids == end_tkn_id
        start_end_mask = torch.bitwise_or(start_tkn_mask, end_tkn_mask)
        index = torch.arange(input_ids.size(1)).repeat(batch_size)[start_end_mask.view(-1)].view(batch_size, -1)
        return mask.scatter(1, index, False)
    
    def get_decoder_batches(self, encode_output: transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions, mask: torch.BoolTensor, pad_idx: int) -> Tuple[torch.Tensor, List[int]]:
        """[summary]

        Args:
            encode_output (transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions): [description]
            mask (torch.BoolTensor): [description]
            model (BertModel): [description]
            pad_idx (int): [description]

        Returns:
            Tuple[torch.Tensor, List[int]]: [description]
        """    
        lengths = mask.sum(1)
        tensors = encode_output.last_hidden_state[mask, :]
        batches = torch.split(tensors, lengths.tolist())
        if lengths.ne(lengths.max()).sum().item() != 0:
            # pad not same length tokens
            tensors_padded = self.pad(batches, lengths.tolist(), pad_idx=pad_idx)
        else:
            # just stack the splitted tensors
            tensors_padded = torch.stack(batches)
        return tensors_padded, lengths.tolist()

    def pad(self, batches: Tuple[torch.Tensor], lengths: List[int], pad_idx: int=1) -> torch.Tensor:
        """Pad for decoder inputs

        Args:
            batches (Tuple[torch.Tensor]): [description]
            lengths (List[int]): [description]
            model (transformers.models.bert.modeling_bert.BertModel): [description]
            pad_idx (int, optional): [description]. Defaults to 1.

        Returns:
            torch.Tensor: [description]
        """       
        padded = []
        max_length = max(lengths)
        for x in batches:
            if len(x) < max_length:
                idxes = self.totensor([pad_idx]*(max_length - len(x)))
                pad_tensor = self.model_bert.embeddings.word_embeddings(idxes)
                padded.append(torch.cat([x, pad_tensor]))
            else:
                padded.append(x)
        return torch.stack(padded)

    def get_sql_answers(self, batch_sqls: List[Dict[str, Any]]):
        """[summary]
        sc: select column
        sa: select agg
        wn: where number
        wc: where column
        wo: where operator
        wv: where value

        Args:
            batch_sqls (List[Dict[str, Any]]): [description]
            tokenizer (KoBertTokenizer): [description]

        Raises:
            EnvironmentError: [description]

        Returns:
            [type]: [description]
        """
        get_ith_element = lambda li, i: [x[i] for x in li]
        g_sc = []
        g_sa = []
        g_wn = []
        g_wc = []
        g_wo = []
        g_wv = []
        for b, sql_dict in enumerate(batch_sqls):
            g_sc.append( sql_dict["sel"] )
            g_sa.append( sql_dict["agg"])

            conds = sql_dict["conds"]
            if not sql_dict["agg"] < 0:
                g_wn.append( len(conds) )
                g_wc.append( get_ith_element(conds, 0) )
                g_wo.append( get_ith_element(conds, 1) )
                g_wv.append( get_ith_element(conds, 2) )
            else:
                raise EnvironmentError

        # get where value tokenized 
        pad_tkn_id = self.tokenizer_bert.pad_token_id
        g_wv_tkns = [[f"{s}{self.hparams.special_end_tkn}" for s in batch_wv] for batch_wv in g_wv]
        g_wv_tkns = [self.tokenizer_bert(batch_wv, add_special_tokens=False)["input_ids"] for batch_wv in g_wv_tkns]
        # add empty list if batch has different where column number
        max_where_cols = max([len(batch_wv) for batch_wv in g_wv_tkns])
        g_wv_tkns = [batch_wv + [[]]*(max_where_cols-len(batch_wv)) if len(batch_wv) < max_where_cols else batch_wv for batch_wv in g_wv_tkns]
        temp = []
        for batch_wv in list(zip(*g_wv_tkns)):
            batch_max_len = max(map(len, batch_wv))
            batch_temp = []
            for wv_tkns in batch_wv:  # iter by number of where clause
                if len(wv_tkns) < batch_max_len:
                    batch_temp.append(wv_tkns + [pad_tkn_id]*(batch_max_len - len(wv_tkns)))
                else:
                    batch_temp.append(wv_tkns)
            temp.append(batch_temp)
        g_wv_tkns = list(zip(*temp))
        
        return g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_tkns
        
        
    def calculate_loss(self, decoder_outputs, gold):
        """
        # Outputs Size
        sc = (B, T_c)
        sa = (B, n_agg_ops)
        wn = (B, 5)
        wc = (B, T_c): binary
        wo = (B, max_where_col_nums, n_cond_ops)
        wv = [(B, T_d_i, vocab_size)] x max_where_col_nums / T_d_i = may have different length for answer
        """
        # Loss Calculation
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_tkns = gold
        # g_wv_tkns = list(zip(*g_wv_tkns))  # (B, where_col_num, T_d_i) -> (where_col_num, B, T_d_i)

        batch_size = decoder_outputs["sc"].size(0)
        loss_sc = self.cross_entropy(decoder_outputs["sc"], self.totensor(g_sc))
        loss_sa = self.cross_entropy(decoder_outputs["sa"], self.totensor(g_sa))
        loss_wn = self.cross_entropy(decoder_outputs["wn"], self.totensor(g_wn)) * self.hparams.wn_penalty

        # need consider: might have different length of where numers
        # So when calculate scores looping by where numbers, ignore the out of length tokens
        loss_wc = 0
        loss_wo = 0
        loss_wv = 0
        for batch_idx, where_num in enumerate(g_wn):  # iter by batch_size: B

            one_hot_dist = torch.zeros_like(decoder_outputs["wc"][batch_idx], device=self.device).scatter(0, self.totensor(g_wc[batch_idx]), 1.0)
            loss_wc += self.binary_cross_entropy(decoder_outputs["wc"][batch_idx], one_hot_dist)

            batch_g_wo = g_wo[batch_idx]  # (where_num_gold,)
            batch_wo = decoder_outputs["wo"][batch_idx, :where_num, :]  # (where_num_predict, n_cond_ops)
            if (len(batch_wo) == 0 and where_num != 0):
                # if predict nothing where clause and answer is not, what loss should be added?
                # simply giving big loss will be enough?
                loss_wo += loss_wn * 100
            else:
                give_wo_penalty = False
                if len(batch_wo) > len(batch_g_wo): 
                    wo_penalty = self.hparams.wo_penalty / 2
                    give_wo_penalty = True
                    batch_wo = batch_wo[:len(batch_g_wo), :]  # (where_num_predict, n_cond_ops)
                elif len(batch_wo) < len(batch_g_wo):
                    # giving penalty if not guessed right where numbers
                    # It becomes problem when reduce the gold tokens but predicted corrected 
                    # Then `loss_wo_base` will be 0, if simply multiply by `loss_wv_base` to loss_base will be zero
                    wo_penalty = self.hparams.wo_penalty
                    give_wo_penalty = True
                    batch_g_wo = batch_g_wo[:len(batch_wo)]  # (where_num_gold,)
                else:
                    wo_penalty = 1.0
                    give_wo_penalty = False
                loss_wo_base = self.cross_entropy(batch_wo, self.totensor(batch_g_wo))
                if give_wo_penalty:
                    loss_wo += loss_wo_base + loss_wn * wo_penalty
                else:
                    loss_wo += loss_wo_base
            
            batch_g_wv = g_wv_tkns[batch_idx][:where_num]  # (where_num_gold, T_d_i)
            batch_wv = [wv[batch_idx] for wv in decoder_outputs["wv"]]  # (where_num_predict, T_d_i, vocab_size)
            if len(batch_wo) == 0 and where_num != 0:
                # if predict nothing where clause and answer is not, what loss should be added?
                loss_wv += loss_wn * 100
                self.pp_wv(1e99)
            else:
                for wv, g_wv_i in zip(batch_wv, batch_g_wv):  # will iter by where_num
                    give_wv_penalty = False
                    if len(wv) > len(g_wv_i):
                        wv_penalty = self.hparams.wo_penalty / 2
                        give_wv_penalty = True
                        wv = wv[:len(g_wv_i), :]  # (T_d_gold, vocab_size)
                    elif len(wv) < len(g_wv_i):
                        # giving penalty if not generate enough tokens
                        # It becomes problem when reduce the gold tokens but predicted corrected 
                        # Then `loss_wv_base` will be 0, if simply multiply by `loss_wv_base` to loss_base will be zero
                        wv_penalty = self.hparams.wo_penalty
                        give_wv_penalty = True
                        g_wv_i = g_wv_i[:len(wv)]  # (T_d_predict,)
                    else:
                        wv_penalty = 1.0
                        give_wv_penalty = False
                    # now have the same T_d size, ignore all over lengthed
                    loss_wv_base = self.cross_entropy(wv, self.totensor(g_wv_i))
                    self.pp_wv(torch.exp(loss_wv_base))
                    if give_wv_penalty:
                        loss_wv += loss_wv_base + loss_wn * wv_penalty
                    else:
                        loss_wv += loss_wv_base
        loss = (loss_sc + loss_sa + loss_wn + loss_wc + loss_wo + loss_wv) / batch_size
        return loss

    def calculate_metrics(self, decoder_outputs, batch_sqls) -> None:
        # Predict tokens
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_tkns = self.get_sql_answers(batch_sqls)
        predicts = self.predict_to_dict(decoder_outputs)
        p_sc, p_sa, p_wn, p_wo = predicts["sc"], predicts["sa"], predicts["wn"], predicts["wo"]
        
        p_wo, g_wo = self.pad_empty_predict_gold(p_wo, g_wo, pad_idx=self.n_cond_ops)  # (B, where_col_num)
        
        acc_sc = self.acc_sc(*map(self.totensor, [p_sc, g_sc]))
        acc_sa = self.acc_sa(*map(self.totensor, [p_sa, g_sa]))
        acc_wn = self.acc_wn(*map(self.totensor, [p_wn, g_wn]))
        
        for batch_idx, where_num in enumerate(g_wn):
            batch_g_wo = g_wo[batch_idx]  # (where_num_gold,)
            batch_wo = p_wo[batch_idx]  # (where_num_predict,)
            acc_wo = self.acc_wo(*map(self.totensor, [batch_wo, batch_g_wo]))
        
    def pad_empty_predict_gold(self, predict, gold, pad_idx):
        res = []
        for p, g in zip(predict, gold):            
            if len(p) < len(g):
                p.extend([pad_idx]*(len(g)-len(p)))
            elif len(p) > len(g):
                g.extend([pad_idx]*(len(p)-len(g)))

            res.append([p, g])

        return list(zip(*res))
    
    def get_batch_data(self, data: List[Dict[str, Any]], table: Dict[str, Dict[str, List[Any]]], start_tkn="[S]", end_tkn="[E]", only_question=False) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """[summary]

        Args:
            data (List[Dict[str, Any]]): [description]
            dbengine (DBEngine): [description]
            start_tkn (str, optional): [description]. Defaults to "[S]".
            end_tkn (str, optional): [description]. Defaults to "[E]".

        Returns:
            Tuple[List[str], List[str], List[Dict[str, Any]]]: [description]
        """    
        batch_qs = [jsonl["question"] for jsonl in data]
        
        tid = [jsonl["table_id"] for jsonl in data]
        batch_ts = []
        for table_id in tid:
            table_str = f"{table_id}" + "".join([
                f"{self.hparams.special_col_tkn}{col}" for col in table[table_id]["header"]
            ])
            # TODO: [EXP] Experiment for generate column directly
            # table_str = f"{start_tkn}{table_id}{end_tkn}" + "".join([
            #     f"{col_tkn}{start_tkn}{col}{end_tkn}" for col in dbengine.schema
            # ]) 
            batch_ts.append(table_str)
        if only_question:
            return batch_qs, batch_ts
        
        batch_sqls = [jsonl["sql"] for jsonl in data]
        return batch_qs, batch_ts, batch_sqls
    
    def compute_all_metrics(self, batch_size):
        acc_sc = self.acc_sc.compute()
        acc_sa = self.acc_sa.compute()
        acc_wn = self.acc_wn.compute()
        acc_wo = self.acc_wo.compute()
        pp_wv = self.pp_wv.compute() / batch_size
        return acc_sc, acc_sa, acc_wn, acc_wo, pp_wv

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch_qs, batch_ts, batch_sqls = self.get_batch_data(batch, self.table, self.hparams.special_start_tkn, self.hparams.special_end_tkn)
        loss, outputs = self(
            batch_qs=batch_qs, 
            batch_ts=batch_ts, 
            batch_sqls=batch_sqls, 
            value_tkn_max_len=None, 
            train=True
        )
        self.calculate_metrics(outputs, batch_sqls)

        acc_sc, acc_sa, acc_wn, acc_wo, pp_wv = self.compute_all_metrics(len(batch))
        self.log("train_step_loss", loss, prog_bar=True, logger=True)
        self.log("train_step_acc_sc", acc_sc, prog_bar=True, logger=True)
        self.log("train_step_acc_sa", acc_sa, prog_bar=True, logger=True)
        self.log("train_step_acc_wn", acc_wn, prog_bar=True, logger=True)
        self.log("train_step_acc_wo", acc_wo, prog_bar=True, logger=True)
        self.log("train_step_pp_wv", pp_wv, prog_bar=True, logger=True)

        return  {"loss": loss}

    def train_epoch_end(self, outputs):
        loss = torch.tensor(0, dtype=torch.float)
        for out in outputs:
            loss += out["loss"].detach().cpu()
        loss = loss / len(outputs)
        
        
        acc_sc, acc_sa, acc_wn, acc_wo, pp_wv = self.compute_all_metrics(len(outputs))
        self.log("train_loss", loss, prog_bar=False, logger=True)
        self.log("train_acc_sc", acc_sc, prog_bar=False, logger=True)
        self.log("train_acc_sa", acc_sa, prog_bar=False, logger=True)
        self.log("train_acc_wn", acc_wn, prog_bar=True, logger=True)
        self.log("train_acc_wo", acc_wo,  prog_bar=False, logger=True)
        self.log("train_pp_wv", pp_wv, prog_bar=False, logger=True)
    
        self.reset_metrics_epoch_end()
        # return {"train_loss": loss, "train_acc_sc": acc_sc, "train_acc_sa": acc_sa, "train_acc_wn": acc_wn, "train_acc_wo": acc_wo, "train_pp_wv": pp_wv}
    
    def validation_step(self, batch, batch_idx):
        batch_qs, batch_ts, batch_sqls = self.get_batch_data(batch, self.table, self.hparams.special_start_tkn, self.hparams.special_end_tkn)
        loss, outputs = self(
            batch_qs=batch_qs, 
            batch_ts=batch_ts, 
            batch_sqls=batch_sqls, 
            value_tkn_max_len=self.hparams.value_tkn_max_len, 
            train=False
        )
        self.calculate_metrics(outputs, batch_sqls)

        return {"loss": loss}
    
    def validation_epoch_end(self, outputs):
        loss = torch.tensor(0, dtype=torch.float)
        for out in outputs:
            loss += out["loss"].detach().cpu()
        loss = loss / len(outputs)

        acc_sc, acc_sa, acc_wn, acc_wo, pp_wv = self.compute_all_metrics(len(outputs))
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc_sc", acc_sc, prog_bar=True, logger=True)
        self.log("val_acc_sa", acc_sa, prog_bar=True, logger=True)
        self.log("val_acc_wn", acc_wn, prog_bar=True, logger=True)
        self.log("val_acc_wo", acc_wo, prog_bar=True, logger=True)
        self.log("val_pp_wv", pp_wv, prog_bar=True, logger=True)
        self.reset_metrics_epoch_end()

    def load_data(self, sql_path: Union[Path, str], table_path: Union[Path, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Load data from path

        Args:
            sql_path (Union[Path, str]): dataset path which contains NL with SQL queries (+answers)
            table_path (Union[Path, str]): table information contains table name, header and values

        Returns:
            Tuple[List[Dict[str, Any]], Dict[str, Any]]: [description]
        """    
        path_sql = Path(sql_path)
        path_table = Path(table_path)

        dataset = self.load_sqls(path_sql)
        table = self.load_tables(path_table)

        return dataset, table
    
    def load_sqls(self, path_sql):
        dataset = []
        with path_sql.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                x = json.loads(line.strip())
                dataset.append(x)
        return dataset
    
    def load_tables(self, path_table):
        table = {}
        with path_table.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                x = json.loads(line.strip())
                table[x['id']] = x
        return table
    
    def create_dataloader(self, mode):
        num_workers = 0 if os.name == "nt" else self.hparams.num_workers
        if mode == "train":
            shuffle = True
            batch_size = self.hparams.train_batch_size
            sql_file = self.hparams.train_sql_file
            table_file = self.hparams.train_table_file
        else:
            shuffle = False
            batch_size = self.hparams.eval_batch_size
            sql_file = self.hparams.eval_sql_file
            table_file = self.hparams.eval_table_file
        
        dataset, self.table = self.load_data(sql_file, table_file)
        
        data_loader = torch.utils.data.DataLoader(
            batch_size=batch_size,
            dataset=dataset,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn # now dictionary values are not merged!
        )
        return data_loader

    def _collate_fn(self, x):
        return x

    def train_dataloader(self):
        return self.create_dataloader(mode="train")

    def val_dataloader(self):
        return self.create_dataloader(mode="eval")
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model_decoder.parameters()),
                                       lr=self.hparams.lr, weight_decay=0)
        opt_bert = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model_bert.parameters()),
                                    lr=self.hparams.lr_bert, weight_decay=0)
        
        optimizers = [opt, opt_bert]
        return optimizers
    
    def predict_to_dict(self, outputs):        
        predicts = {}
        predicts["sc"] = self.model_decoder.predict_decoder("sc", select_outputs=outputs["sc"])
        predicts["sa"] = self.model_decoder.predict_decoder("sa", agg_outputs=outputs["sa"])
        predicts["wn"] = self.model_decoder.predict_decoder("wn", where_num_outputs=outputs["wn"])
        predicts["wc"] = self.model_decoder.predict_decoder("wc", where_col_outputs=outputs["wc"], where_nums=predicts["wn"])
        predicts["wo"] = self.model_decoder.predict_decoder("wo", where_op_outputs=outputs["wo"], where_nums=predicts["wn"])
        predicts["wv_tkns"] = self.model_decoder.predict_decoder("wv", where_value_outputs=outputs["wv"])  # (B, value_tkn_max_len) x where_nums
        # internally wv means wv_tkns, will convert to string here using tokenizer
        predicts["wv"] = []
        for where_idx, wv_tkns in enumerate(predicts["wv_tkns"]): # iter: (B, value_tkn_max_len)
            predicts["wv"].append([self.tokenizer_bert.decode(self.totensor(batch_wv)) for batch_wv in wv_tkns])
                
        predicts["wv"] = list(zip(*predicts["wv"]))
        
        return predicts

    def predict_outputs(self, data, table):
        batch_qs, batch_ts = self.get_batch_data(data, table, only_question=True)
        outputs = self.forward_outputs(batch_qs, batch_ts, batch_sqls=None, value_tkn_max_len=self.hparams.value_tkn_max_len, train=False)
        return self.predict_to_dict(outputs)