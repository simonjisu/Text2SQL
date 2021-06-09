import torch
import transformers
import pytorch_lightning as pl
from pathlib import Path
from .model import Text2SQL
torch.multiprocessing.set_sharing_strategy("file_system")

def train(args):
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Transformers Version: {transformers.__version__}")
    print(f"PyTorch Lightning Version: {pl.__version__}")

    # Path Check
    if not Path(args.log_dir).exists():
        Path(args.log_dir, parent=True)

    if not Path(args.ckpt_dir).exists():
        Path(args.ckpt_dir, parent=True)

    args_dict = dict(
        db_path = args.db_path,
        model_bert_path = args.model_bert_path,
        # Dataloader
        train_sql_file = args.train_sql_file,
        train_table_file = args.train_table_file,
        train_batch_size = args.train_batch_size,
        eval_sql_file = args.eval_sql_file,
        eval_table_file = args.eval_table_file,
        eval_batch_size = args.eval_batch_size,
        num_workers = args.num_workers,
        # Model-decoder
        hidden_size = args.hidden_size,
        num_layers = args.num_layers,
        dropout_ratio = args.dropout_ratio,
        max_where_conds = args.max_where_conds,
        value_tkn_max_len = args.value_tkn_max_len, 
        # Tokenizer
        special_start_tkn = args.special_start_tkn, 
        special_end_tkn = args.special_end_tkn,
        special_col_tkn = args.special_col_tkn,
        # Loss Function
        wn_penalty = args.wn_penalty,  # scale up for guessing where number
        wo_penalty = args.wo_penalty,
        wv_penalty = args.wv_penalty,  # giving penalty if not generate enough tokens, if generate more than answer lenght will give 1/2 of it
        # Optimizer
        lr = args.lr,
        lr_bert = args.lr_bert
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="epoch{epoch:02d}-{val_loss:.3f}-{val_acc_sc:.3f}-{val_acc_sa:.3f}-{val_acc_wn:.3f}-{val_acc_wo:.3f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
    )
    tb_logger = pl.loggers.TensorBoardLogger(Path(args.log_dir), name=args.task)
    # mlf_logger = pl.loggers.MLFlowLogger(tracking_uri=f"file:{args.log_dir}", experiment_name=args.task+"_mlf")
    earlystop_callback = pl.callbacks.EarlyStopping("val_loss", mode="min")
    pl.seed_everything(args.seed)
    model = Text2SQL(**args_dict)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, earlystop_callback],
        max_epochs=args.num_train,
        deterministic=torch.cuda.is_available(),
        gpus=-1 if torch.cuda.is_available() else None,
        num_sanity_val_steps=0,
        logger=tb_logger#[tb_logger, mlf_logger]
    )
    trainer.fit(model)