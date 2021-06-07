from KoBertTokenizer import KoBertTokenizer
from transformers import BertModel, BertConfig

def get_bert(model_path: str, device: str, max_col_length: int, output_hidden_states: bool=False):
    
    special_tokens = ["[S]", "[E]", "[COL]"] # sequence start, sequence end, column tokens
    tokenizer = KoBertTokenizer.from_pretrained(model_path, add_special_tokens=True, additional_special_tokens=special_tokens)
    config = BertConfig.from_pretrained(model_path)
    config.output_hidden_states = output_hidden_states
    
    model = BertModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.output_hidden_states = output_hidden_states
    model.to(device)
    
    return model, tokenizer, config