import os
proj_dir=os.getcwd()

seed=42

# Data
dataset_name = 'sciq' # huggingface dataset name
column_renaming = {"support":"context", "correct_answer": "answer"}
train_file = f'{proj_dir}/mrqa/preprocessed_train.csv'
dev_file = f'{proj_dir}/mrqa/preprocessed_dev.csv'
dev_from_train=.05 ## set to -1 if use dev file for validation, else subsample from train
filter_long_sequences=True

# debug
debug=False
tiny=False
tiny_size=1024
overfit=False

# Task Type
use_context=True
shots=-1

# Logging Arguments
run_name = "t5-small-open-book"
report_to='wandb'

# Model and Tokenizer
model_name = 't5-small'
input_max_length = 256
output_max_length = 32
num_workers=4
learning_rate='auto'

#optimizer
"""
Adafactor, AdamW, Adam
"""
optimizer = 'Adafactor' 


# Checkpoint arguments
save_top_k=3
monitor='val_loss'
mode='min'

# Dataloader Arguments
batch_size=128

# Generation Arguments
# from https://github.com/ellaneeman/disent_qa/blob/main/config.json
repetition_penalty=2.5
length_penalty=1.0
no_repeat_ngram_size=3
num_beams=1
early_stopping=True


# Trainer Arguments
accelerator='auto'
strategy='ddp'
devices=2
num_nodes=1
""" 
(Union[Literal[64, 32, 16], Literal[‘16-mixed’, ‘bf16-mixed’, ‘32-true’, 
‘64-true’], Literal[‘64’, ‘32’, ‘16’, ‘bf16’]]) – 
Double precision ("64"), full precision ("32"), half precision AMP ("16-mixed"), 
or bfloat16 precision AMP ("bf16-mixed").
"""
precision="bf16-mixed"
fast_dev_run=False
max_epochs=1
min_epochs=1
max_steps=-1
min_steps=None
max_time=None
limit_train_batches=None
limit_val_batches=None
limit_test_batches=None
limit_predict_batches=None
overfit_batches=0.0
val_check_interval=800
check_val_every_n_epoch=1
num_sanity_val_steps=0
log_every_n_steps=200
enable_progress_bar=True
enable_model_summary=None
accumulate_grad_batches=1
gradient_clip_val=None
gradient_clip_algorithm=None
deterministic=None
benchmark=None
inference_mode=True
use_distributed_sampler=True
profiler=None
detect_anomaly=False
barebones=False
sync_batchnorm=True
reload_dataloaders_every_n_epochs=0