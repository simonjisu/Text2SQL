__author__ = "simonjisu"
import os
import json

import torch
import torch.nn as nn

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
        self.totensor_cpu = lambda x: torch.LongTensor(x)
        self.table_dict = {"train": None, "eval": None}
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
        
    def forward(self, batch_qs, batch_ts, batch_sqls=None, value_tkn_max_len=None, train=True):
        outputs, _ = self.forward_outputs(batch_qs, batch_ts, batch_sqls, value_tkn_max_len, train)
        g_sc, g_sa, g_wn, g_wc, g_wo, _, g_wv_tkns = self.get_sql_answers(batch_sqls)
        gold = [g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_tkns]
        loss = self.calculate_loss(outputs, gold)  # when calculate loss must need gold answer
        metrics = self.calculate_metrics(outputs, gold)
        return loss, metrics
            
    def forward_outputs(self, batch_qs, batch_ts, batch_sqls=None, value_tkn_max_len=None, train=True, rt_attn=False):
        # --- Get Answer & Variables ---
        if train:
            assert value_tkn_max_len is None, "In train phase, `value_tkn_max_len` must be None"
            assert batch_sqls is not None, "In train phase, `batch_sqls` must not be None"
            g_sc, g_sa, g_wn, g_wc, g_wo, _, g_wv_tkns = self.get_sql_answers(batch_sqls)
            gold = [g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_tkns]
        else:
            assert value_tkn_max_len is not None, "In test Phase, `value_tkn_max_len` must not be None"
            gold = None
            value_tkn_max_len = value_tkn_max_len
            
        # --- Get Inputs for Encoder --- 
        encode_inputs = self.tokenizer_bert(
            batch_qs, batch_ts, 
            max_length=self.hparams.max_length, padding=True, truncation=True, return_tensors="pt", 
            return_attention_mask=True, 
            return_special_tokens_mask=False, 
        ).to(self.device)  # encode_input doesn't return the cuda device
        
        # --- Forward Encoder ---
        encode_outputs = self.model_bert(**encode_inputs, output_attentions=rt_attn)
        
        # --- Get Inputs for Decoder ---
        input_question_mask, input_db_mask, input_col_mask = self.get_input_mask(encode_inputs)
        question_padded, question_lengths = self.get_decoder_batches(encode_outputs, input_question_mask, pad_idx=self.tokenizer_bert.pad_token_id)
        db_padded, _ = self.get_decoder_batches(encode_outputs, input_db_mask, pad_idx=self.tokenizer_bert.pad_token_id)
        col_padded, col_lengths = self.get_decoder_batches(encode_outputs, input_col_mask, pad_idx=self.tokenizer_bert.pad_token_id)
        
        # --- Forward Decoder ---
        decoder_outputs, decoder_attns = self.model_decoder(
            question_padded, 
            db_padded, 
            col_padded, 
            question_lengths, 
            col_lengths, 
            value_tkn_max_len, 
            gold,
            rt_attn
        )
        
        attns = {
            "encoder": encode_outputs.attentions,
            "decoder": decoder_attns
        }
        
        return decoder_outputs, attns

    def get_input_mask(self, encode_inputs):
        sep_tkn_mask = encode_inputs["input_ids"] == self.tokenizer_bert.sep_token_id
        col_tkn_id = self.tokenizer_bert.additional_special_tokens_ids[2]

        input_question_mask = torch.bitwise_and(encode_inputs["token_type_ids"] == 0, encode_inputs["attention_mask"].bool())
        input_question_mask = torch.bitwise_and(input_question_mask, ~sep_tkn_mask) # [SEP] mask out
        input_question_mask[:, 0] = False  # [CLS] mask out

        db_mask = torch.bitwise_and(encode_inputs["token_type_ids"] == 1, encode_inputs["attention_mask"].bool())
        col_tkn_mask = encode_inputs["input_ids"] == col_tkn_id
        db_mask = torch.bitwise_and(db_mask, ~col_tkn_mask)
        db_mask = torch.bitwise_xor(db_mask, torch.bitwise_and(sep_tkn_mask, encode_inputs["token_type_ids"] == 1))
        
        return input_question_mask, db_mask, col_tkn_mask

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
        if lengths.ne(lengths.max()).sum() != 0:
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
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = [], [], [], [], [], []
        for sql_dict in batch_sqls:
            g_sc.append( sql_dict["sel"] )
            g_sa.append( sql_dict["agg"] )

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
        g_wv_tkns = [self.tokenizer_bert(batch_wv, add_special_tokens=False)["input_ids"] if len(batch_wv) > 0 else batch_wv for batch_wv in g_wv_tkns]
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
        g_wv_tkns = list(map(list, g_wv_tkns))
        return g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_tkns

    def _flatten(self, li):
        return [x for inner_li in li for x in inner_li] 

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
        # Loss: sc, sa, wn
        batch_size = decoder_outputs["sc"].size(0)
        loss_sc = self.cross_entropy(decoder_outputs["sc"], self.totensor(g_sc))
        loss_sa = self.cross_entropy(decoder_outputs["sa"], self.totensor(g_sa))
        loss_wn = self.cross_entropy(decoder_outputs["wn"], self.totensor(g_wn)) * self.hparams.wn_penalty

        # Loss: wc, wo, wv
        # need consider: might have different length of where numers
        # So when calculate scores looping by where numbers, ignore the out of length tokens
        loss_wc = 0
        loss_wo = 0
        loss_wv = 0
        for batch_idx, where_num in enumerate(g_wn):  # iter by batch_size: B
            # Loss: wc
            one_hot_dist = torch.zeros_like(decoder_outputs["wc"][batch_idx]).scatter(0, self.totensor(g_wc[batch_idx]), 1.0)
            loss_wc += self.binary_cross_entropy(decoder_outputs["wc"][batch_idx], one_hot_dist)

            # Loss: wo
            batch_g_wo = g_wo[batch_idx]  # (where_num,)
            batch_wo = decoder_outputs["wo"][batch_idx, :where_num, :]  # (where_num, n_cond_ops)
            loss_wo += self.cross_entropy(batch_wo, self.totensor(batch_g_wo))

            # Loss: wv
            batch_g_wv = g_wv_tkns[batch_idx][:where_num]  # (where_num, T_d_i)
            batch_wv = [wv[batch_idx] for wv in decoder_outputs["wv"]]  # (where_num, T_d_i, vocab_size)
            for wv, g_wv_i in zip(batch_wv, batch_g_wv):
                if len(wv) > len(g_wv_i):
                    g_wv_i = g_wv_i + [self.tokenizer_bert.pad_token_id]*(len(wv) - len(g_wv_i))
                elif len(wv) < len(g_wv_i):
                    pad_tkns = self.model_decoder.where_value_decoder.embedding_layer(torch.LongTensor([self.tokenizer_bert.pad_token_id]*(len(g_wv_i) - len(wv))).to(self.device))
                    pad_tkns_res = self.model_decoder.where_value_decoder.output_linear(pad_tkns)
                    wv = torch.cat([wv, pad_tkns_res])
                loss_wv += self.cross_entropy_wv(wv, self.totensor(g_wv_i))

        self.pp_wv.update(torch.exp(loss_wv / batch_size))     
        loss = (loss_sc + loss_sa + loss_wn + loss_wc + loss_wo + loss_wv) / batch_size
        return loss

    def create_metricollection(self, prefix, num_classes=None, ignore_index=None):
        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(num_classes=num_classes, ignore_index=ignore_index), 
            torchmetrics.Precision(num_classes=num_classes, ignore_index=ignore_index), 
            torchmetrics.Recall(num_classes=num_classes, ignore_index=ignore_index)
        ])
        return metrics.clone(prefix=prefix)

    def create_metrics(self):
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")
        self.binary_cross_entropy = nn.BCEWithLogitsLoss(reduction="sum")
        self.cross_entropy_wv = nn.CrossEntropyLoss(reduction="sum", ignore_index=1)

        self.metric_train_sc = self.create_metricollection(prefix="train_sc_")
        self.metric_valid_sc = self.create_metricollection(prefix="val_sc_")

        self.metric_train_sa = self.create_metricollection(prefix="train_sa_", num_classes=self.n_agg_ops)
        self.metric_valid_sa = self.create_metricollection(prefix="val_sa_", num_classes=self.n_agg_ops)

        self.metric_train_wn = self.create_metricollection(prefix="train_wn_", num_classes=self.hparams.max_where_conds+1)
        self.metric_valid_wn = self.create_metricollection(prefix="val_wn_", num_classes=self.hparams.max_where_conds+1)

        self.metric_train_wc = self.create_metricollection(prefix="train_wc_")
        self.metric_valid_wc = self.create_metricollection(prefix="val_wc_")

        self.metric_train_wo = self.create_metricollection(prefix="train_wo_", num_classes=self.n_cond_ops+1)
        self.metric_valid_wo = self.create_metricollection(prefix="val_wo_", num_classes=self.n_cond_ops+1)

        self.metric_train_wv = self.create_metricollection(prefix="train_wv_", num_classes=self.vocab_size, ignore_index=self.tokenizer_bert.pad_token_id)
        self.metric_valid_wv = self.create_metricollection(prefix="val_wv_", num_classes=self.vocab_size, ignore_index=self.tokenizer_bert.pad_token_id)

        self.pp_wv = Perplexity()

    def reset_metrics_epoch_end(self, train=True):
        """will be automatically called and epoch_end"""
        if train:
            self.metric_train_sc.reset()
            self.metric_train_sa.reset()
            self.metric_train_wn.reset()
            self.metric_train_wc.reset()
            self.metric_train_wo.reset()
            self.metric_train_wv.reset()
        else:
            self.metric_valid_sc.reset()
            self.metric_valid_sa.reset()
            self.metric_valid_wn.reset()
            self.metric_valid_wc.reset()
            self.metric_valid_wo.reset()
            self.metric_valid_wv.reset()
        
        self.pp_wv.reset()

    def calculate_metrics(self, decoder_outputs, gold) -> None:
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_tkns = gold # self.get_sql_answers(batch_sqls)
        metrics = {
            "predict": {"sc": None, "sa": None, "wn": None, "wc": [], "wo": [], "wv": []},
            "gold": {"sc": g_sc, "sa": g_sa, "wn": g_wn, "wc": [], "wo": [], "wv": []},
        }
        # Predict tokens
        predicts = self.predict_to_dict(decoder_outputs)
        p_sc, p_sa, p_wn, p_wc, p_wo, p_wv_tkns = predicts["sc"], predicts["sa"], predicts["wn"], predicts["wc"], predicts["wo"], predicts["wv_tkns"]
        
        # Where Column
        p_wc, g_wc = self.pad_metrics(p_wc, g_wc, pad_idx=999)
        p_wo, g_wo = self.pad_metrics(p_wo, g_wo, pad_idx=self.n_cond_ops)
        # select the where number first: (B, gold_where_nums, T_d)
        p_wv_tkns = list(map(list, zip(*p_wv_tkns)))
        p_wv_tkns, g_wv_tkns = self.pad_metrics(p_wv_tkns, g_wv_tkns, pad_idx=self.tokenizer_bert.pad_token_id)

        metrics["predict"]["sc"] = p_sc
        metrics["predict"]["sa"] = p_sa
        metrics["predict"]["wn"] = p_wn

        for batch_idx in range(len(g_sc)):
            batch_wc = p_wc[batch_idx]
            batch_g_wc = g_wc[batch_idx]
            metrics["predict"]["wc"].extend(batch_wc)
            metrics["gold"]["wc"].extend(batch_g_wc)

            batch_wo = p_wo[batch_idx]
            batch_g_wo = g_wo[batch_idx]
            metrics["predict"]["wo"].extend(batch_wo)
            metrics["gold"]["wo"].extend(batch_g_wo)

            batch_g_wv_tkns = g_wv_tkns[batch_idx]
            batch_wv_tkns = p_wv_tkns[batch_idx]
            batch_wv_tkns, batch_g_wv_tkns = self.pad_metrics(
                batch_wv_tkns, batch_g_wv_tkns, pad_idx=self.tokenizer_bert.pad_token_id
            ) # each contains: (where_num, T_d_g)
            batch_wv_tkns = self._flatten(batch_wv_tkns)
            batch_g_wv_tkns = self._flatten(batch_g_wv_tkns)
            metrics["predict"]["wv"].extend(batch_wv_tkns)
            metrics["gold"]["wv"].extend(batch_g_wv_tkns)

        metrics_tensor = {}
        metrics_tensor["predict"] = {k: self.totensor_cpu(v) for k, v in metrics["predict"].items()}
        metrics_tensor["gold"] = {k: self.totensor_cpu(v) for k, v in metrics["gold"].items()}
        return metrics_tensor
    
    def pad_metrics(self, predict, gold, pad_idx=999):
        """
        predict, gold = (B, where_num) or (where_num, T_d_i)
        Case1 len(p) > len(g) 
            -> just get len(g)
            -> if g is empty, pad g to len(p)
        Case2 len(p) < len(g) 
            -> pad p
        Case3 len(p) = len(g) 
            -> if p & g is empty, pad only 1 with p & g
        """
        res = []
        for p, g in zip(predict, gold):
            if len(p) > len(g):
                if len(g) == 0:
                    g = [pad_idx] * len(p)
                else:
                    p = p[:len(g)]
            elif len(p) < len(g):
                p.extend([pad_idx]*(len(g)-len(p)))
            else:
                if len(p) == 0 and len(g) == 0:
                    g = [pad_idx]
                    p = [pad_idx]
            res.append([p, g])

        return tuple(map(list, zip(*res)))
    
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

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch_qs, batch_ts, batch_sqls = self.get_batch_data(batch, self.table_dict["train"], self.hparams.special_start_tkn, self.hparams.special_end_tkn)
        loss, metrics = self(
            batch_qs=batch_qs, 
            batch_ts=batch_ts, 
            batch_sqls=batch_sqls, 
            value_tkn_max_len=None, 
            train=True
        )
        self.log("tr_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        return {"loss": loss, "metrics": metrics}

    def train_epoch_end(self, outputs):
        loss = 0 # torch.tensor(0, dtype=torch.float, device=self.device)
        for out in outputs:
            loss += out["loss"].detach()
            m = out["metrics"]

            self.metric_train_sc(m["predict"]["sc"], m["gold"]["sc"])
            self.metric_train_sa(m["predict"]["sa"], m["gold"]["sa"])
            self.metric_train_wn(m["predict"]["wn"], m["gold"]["wn"])
            self.metric_train_wc(m["predict"]["wc"], m["gold"]["wc"])
            self.metric_train_wo(m["predict"]["wo"], m["gold"]["wo"])
            self.metric_train_wv(m["predict"]["wv"], m["gold"]["wv"])

        loss = loss / len(outputs)
        pp = self.pp_wv.compute() / len(outputs)

        train_sc = self.metric_train_sc.compute()
        train_sa = self.metric_train_sa.compute()
        train_wn = self.metric_train_wn.compute()
        train_wc = self.metric_train_wc.compute()
        train_wo = self.metric_train_wo.compute()
        train_wv = self.metric_train_wv.compute()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(train_sc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(train_sa, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(train_wn, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(train_wc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(train_wo, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(train_wv, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train_pp_wv", pp, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.reset_metrics_epoch_end(train=True)
    
    def validation_step(self, batch, batch_idx):
        batch_qs, batch_ts, batch_sqls = self.get_batch_data(batch, self.table_dict["eval"], self.hparams.special_start_tkn, self.hparams.special_end_tkn)
        loss, metrics = self(
            batch_qs=batch_qs, 
            batch_ts=batch_ts, 
            batch_sqls=batch_sqls, 
            value_tkn_max_len=self.hparams.value_tkn_max_len, 
            train=False
        )

        return {"loss": loss, "metrics": metrics}

    def validation_epoch_end(self, outputs):
        loss = 0 # torch.tensor(0, dtype=torch.float, device=self.device)
        for out in outputs:
            loss += out["loss"].detach()
            m = out["metrics"]

            self.metric_valid_sc(m["predict"]["sc"], m["gold"]["sc"])
            self.metric_valid_sa(m["predict"]["sa"], m["gold"]["sa"])
            self.metric_valid_wn(m["predict"]["wn"], m["gold"]["wn"])
            self.metric_valid_wc(m["predict"]["wc"], m["gold"]["wc"])
            self.metric_valid_wo(m["predict"]["wo"], m["gold"]["wo"])
            self.metric_valid_wv(m["predict"]["wv"], m["gold"]["wv"])

        loss = loss / len(outputs)
        pp = self.pp_wv.compute() / len(outputs)

        val_sc = self.metric_valid_sc.compute()
        val_sa = self.metric_valid_sa.compute()
        val_wn = self.metric_valid_wn.compute()
        val_wc = self.metric_valid_wc.compute()
        val_wo = self.metric_valid_wo.compute()
        val_wv = self.metric_valid_wv.compute()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(val_sc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(val_sa, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(val_wn, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(val_wc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(val_wo, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(val_wv, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_pp_wv", pp, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.reset_metrics_epoch_end(train=False)

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
            for line in f:
                x = json.loads(line.strip())
                dataset.append(x)
        return dataset
    
    def load_tables(self, path_table):
        table = {}
        with path_table.open("r", encoding="utf-8") as f:
            for line in f:
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
        
        dataset, table = self.load_data(sql_file, table_file)
        self.table_dict[mode] = table
        data_loader = torch.utils.data.DataLoader(
            batch_size=batch_size,
            dataset=dataset,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False if num_workers > 0 else False,
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
        for wv_tkns in predicts["wv_tkns"]: # iter: (B, value_tkn_max_len)
            predicts["wv"].append([self.tokenizer_bert.decode(self.totensor(batch_wv)) for batch_wv in wv_tkns])
                
        predicts["wv"] = list(zip(*predicts["wv"]))
        
        return predicts

    def predict_outputs(self, data, table, rt_attn=False):
        batch_qs, batch_ts = self.get_batch_data(data, table, only_question=True)
        outputs, attns = self.forward_outputs(batch_qs, batch_ts, batch_sqls=None, value_tkn_max_len=self.hparams.value_tkn_max_len, train=False, rt_attn=rt_attn)
        return self.predict_to_dict(outputs), attns