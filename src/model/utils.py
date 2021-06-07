import json
import torch
from pathlib import Path
from dbengine import DBEngine
from torch.utils.data import DataLoader
from KoBertTokenizer import KoBertTokenizer
import transformers
from typing import List, Dict, Any, Tuple, Union

# [Data Part]

def load_data(sql_path: Union[Path, str], table_path: Union[Path, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load data from path

    Args:
        sql_path (Union[Path, str]): dataset path which contains NL with SQL queries (+answers)
        table_path (Union[Path, str]): table information contains table name, header and values

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]: [description]
    """    
    path_sql = Path(sql_path)
    path_table = Path(table_path)

    dataset = []
    table = {}
    with path_sql.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            x = json.loads(line.strip())
            dataset.append(x)

    with path_table.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            x = json.loads(line.strip())
            table[x['id']] = x
            
    return dataset, table

def get_data_loader(dataset: List[Dict[str, Any]], batch_size: int, num_workers: int, train: bool=True) -> DataLoader:
    """[summary]

    Args:
        dataset (List[Dict[str, Any]]): [description]
        batch_size (int): [description]
        num_workers (int): [description]
        train (bool, optional): [description]. Defaults to True.

    Returns:
        DataLoader: [description]
    """    
    data_loader = DataLoader(
        batch_size=2,
        dataset=dataset,
        shuffle=True if train else False,
        num_workers=num_workers,
        collate_fn=lambda x: x # now dictionary values are not merged!
    )
    return data_loader

def get_batch_data(data: Dict[str, Any], dbengine: DBEngine, start_tkn="[S]", end_tkn="[E]") -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """[summary]

    Args:
        data (Dict[str, Any]): [description]
        dbengine (DBEngine): [description]
        start_tkn (str, optional): [description]. Defaults to "[S]".
        end_tkn (str, optional): [description]. Defaults to "[E]".

    Returns:
        Tuple[List[str], List[str], List[Dict[str, Any]]]: [description]
    """    
    batch_qs = [jsonl["question"] for jsonl in data]
    tid = [jsonl["table_id"] for jsonl in data]
    batch_sqls = [jsonl["sql"] for jsonl in data]
    batch_ts = []
    for table_id in tid:
        dbengine.get_schema_info(table_id)
        table_str = f"{table_id}" + "".join([
            f"[COL]{col}" for col in dbengine.schema
        ])
        # TODO: [EXP] Experiment for generate column directly
        # table_str = f"{start_tkn}{table_id}{end_tkn}" + "".join([
        #     f"[COL]{start_tkn}{col}{end_tkn}" for col in dbengine.schema
        # ]) 
        batch_ts.append(table_str)
    
    return batch_qs, batch_ts, batch_sqls

# [Input for Decoder]
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


def get_decoder_input_mask(input_ids: torch.Tensor, mask: torch.BoolTensor, batch_size: int, start_tkn_id: int, end_tkn_id: int) -> torch.BoolTensor:
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


def get_input_mask_and_answer(encode_input: transformers.tokenization_utils_base.BatchEncoding, tokenizer: KoBertTokenizer) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
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
    input_idx = torch.arange(max_length).repeat(batch_size, 1)
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
        lambda x: torch.stack(x), 
        zip(*[torch.split(x, size.tolist()) for x, size in zip(table_header_tkn_idx.view(batch_size, -1), split_size)])
    )

    table_tkn_idx = table_tkn_idx[:, 1:]

    # TODO: [EXP] Experiment for generate column directly
    # If [EXP], Mask include [S] & [E] tokens
    table_tkn_mask = torch.zeros_like(encode_input["input_ids"], dtype=torch.bool).scatter(1, table_tkn_idx, True)
    header_tkn_mask = torch.zeros_like(encode_input["input_ids"], dtype=torch.bool).scatter(1, header_tkn_idx, True)

    # TODO: [EXP] Experiment for generate column directly
    # For Decoder Input, Maskout [S], [E] for table & header -> will be done automatically
    input_table_mask = get_decoder_input_mask(
        encode_input["input_ids"], table_tkn_mask, batch_size, start_tkn_id, end_tkn_id
    )
    input_header_mask = get_decoder_input_mask(
        encode_input["input_ids"], header_tkn_mask, batch_size, start_tkn_id, end_tkn_id
    )

    # [COL] token mask: this is for attention
    col_tkn_idx = input_idx[col_tkn_mask].view(batch_size, -1)
    input_col_mask = torch.zeros_like(encode_input["input_ids"], dtype=torch.bool).scatter(1, col_tkn_idx, True)

    # TODO: [EXP] Experiment for generate column directly
    # For Answer, Maskout [S] for table & header 
    # answer_table_tkns = get_answer(
    #     encode_input["input_ids"], table_tkn_mask, batch_size, start_tkn_id, end_tkn_id
    # )
    # answer_header_tkns = get_answer(
    #     encode_input["input_ids"], header_tkn_mask, batch_size, start_tkn_id, end_tkn_id
    # )
    
    return input_question_mask, input_table_mask, input_header_mask, input_col_mask # , answer_table_tkns, answer_header_tkns


## Pad for decoder inputs
def pad(batches: Tuple[torch.Tensor], lengths: List[int], model: transformers.models.bert.modeling_bert.BertModel, pad_idx: int=1) -> torch.Tensor:
    """[summary]

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
            pad_tensor = model.embeddings.word_embeddings(torch.LongTensor([pad_idx]*(max_length - len(x))))
            padded.append(torch.cat([x, pad_tensor]))
        else:
            padded.append(x)
    return torch.stack(padded)

def get_decoder_batches(encode_output: transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions, mask: torch.BoolTensor, model: BertModel, pad_idx: int) -> Tuple[torch.Tensor, List[int]]:
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
        tensors_padded = pad(batches, lengths.tolist(), model, pad_idx=pad_idx)
    else:
        # just stack the splitted tensors
        tensors_padded = torch.stack(batches)
    return tensors_padded, lengths.tolist()

def get_pad_mask(lengths: List[int]) -> torch.Tensor:
    """[summary]

    Args:
        lengths (List[int]): [description]

    Returns:
        torch.Tensor: [description]
    """    
    batch_size = len(lengths)
    max_len = max(lengths)
    mask = torch.ones(batch_size, max_len)
    for i, l in enumerate(lengths):
        mask[i, :l] = 0
    return mask

# [Create Answer(Gold)]
def get_sql_answers(batch_sqls: List[Dict[str, Any]], tokenizer: KoBertTokenizer, end_tkn_in_tokenzier_idx:int=1):
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
        end_tkn_in_tokenzier_idx (int, optional): [description]. Defaults to 1.

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
    end_tkn = tokenizer.additional_special_tokens[end_tkn_in_tokenzier_idx]
    pad_tkn_id = tokenizer.pad_token_id
    g_wv_tkns = [[f"{s}{end_tkn}" for s in batch_wv] for batch_wv in g_wv]
    g_wv_tkns = [tokenizer(batch_wv, add_special_tokens=False)["input_ids"] for batch_wv in g_wv_tkns]
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