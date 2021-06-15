import argparse
from src.trainer import train

def argument_parsing(preparse=False):
    parser = argparse.ArgumentParser(description="Text2SQL Argparser",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path & Dataloader
    parser.add_argument("-dbp", "--db_path", type=str, default="./src/data/samsung_new.db",
                   help="Path for database with *.db filename")
    parser.add_argument("-bertp", "--model_bert_path", type=str, default="monologg/kobert",
                   help="Path for BERT model")
    parser.add_argument("-trainsql", "--train_sql_file", type=str, default="./src/data/NLSQL_train.jsonl",
                   help="Path for train sql path with *.jsonl filename")
    parser.add_argument("-traintable", "--train_table_file", type=str, default="./src/data/table_train.jsonl",
                   help="Path for train table path with *.jsonl filename")
    parser.add_argument("-evalsql", "--eval_sql_file", type=str, default="./src/data/NLSQL_test.jsonl",
                   help="Path for eval sql path with *.jsonl filename")
    parser.add_argument("-evaltable", "--eval_table_file", type=str, default="./src/data/table_test.jsonl",
                   help="Path for eval table path with *.jsonl filename")
    parser.add_argument("-logp", "--log_dir", type=str, default="./logs",
                   help="Record Path")
    parser.add_argument("-ckptp", "--ckpt_dir", type=str, default="./ckpt",
                   help="Checkpoint Path")
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=16,
                   help="Batch size for train loader")
    parser.add_argument("-ebs", "--eval_batch_size", type=int, default=16,
                    help="Batch size for validation loader")
    parser.add_argument("-nwks", "--num_workers", type=int, default=4,
                    help="Number of workers for loader")

    # Decoder
    parser.add_argument("-dhs", "--hidden_size", type=int, default=100,
                   help="Decoder hidden size")
    parser.add_argument("-dnl", "--num_layers", type=int, default=2,
                   help="Decoder number of layers")
    parser.add_argument("-ddrp", "--dropout_ratio", type=float, default=0.3,
                   help="Decoder dropout ratio")
    parser.add_argument("-dmwc", "--max_where_conds", type=int, default=4,
                   help="Decoder maximum where conditions that can guess")
    parser.add_argument("-dvtml", "--value_tkn_max_len", type=int, default=20,
                   help="Decoder maximum tokens that where clause value can be generated")
    # Tokenizer
    parser.add_argument("-tstkn", "--special_start_tkn", type=str, default="[S]",
                   help="Tokenizer for start special token")
    parser.add_argument("-tetkn", "--special_end_tkn", type=str, default="[E]",
                   help="Tokenizer for start special token")
    parser.add_argument("-tctkn", "--special_col_tkn", type=str, default="[COL]",
                   help="Tokenizer for start special token")
    # Loss Function
    parser.add_argument("-pwn", "--wn_penalty", type=float, default=2.0,
                   help="Penalty for where number")
    parser.add_argument("-pwo", "--wo_penalty", type=float, default=4.0,
                   help="Penalty for where operator")
    parser.add_argument("-pwv", "--wv_penalty", type=float, default=5.0,
                   help="Penalty for where value")               
    # Optimizer
    parser.add_argument("-nt", "--num_train", type=int, default=10,
                   help="Number of training epochs")
    parser.add_argument("-lr", "--lr", type=float, default=1e-3,
                   help="Decoder learning rate")
    parser.add_argument("-lrb", "--lr_bert", type=float, default=1e-5,
                   help="BERT learning rate")  
    # Seed
    parser.add_argument("-sd", "--seed", type=int, default=88,
                   help="Seed Number")

    # Records 
    parser.add_argument("-tsk", "--task", type=str, default="TEXT2SQL_v1",
                   help="Record Task")
    parser.add_argument("-logevery", "--log_every_n_steps", type=int, default=25,
                   help="Log every n steps")
    # GPU Setting
    parser.add_argument("-ngpus", "--num_gpus", type=int, default=-1,
                   help="Number of gpus")
    if preparse:
        return parser
    
    args = parser.parse_args()
    return args


def main():
    args = argument_parsing()
    train(args)

if __name__ == "__main__":
    main()