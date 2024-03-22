import os
proj_dir=os.getcwd()

seed=42
debug=True
strategy='ddp'

prefix_tuning=False
prefix_tokens=30

# Output directory
output_dir = f'{os.getenv("OUTPUT_DIR", f"{proj_dir}/output")}/llama2/lora/natural_instructions_200k'
resume_from_checkpoint = None
metrics = ['bleu']

raw_data =  "Muennighoff/natural-instructions"


preprocessing_kwargs = {
    "remove_html": False,
    "pad_punctuation": False,
    "drop_tables": False,
    "column_dict": {"inputs": ["definition", "inputs"], "target": "targets"},
    "input_template": "[INST] {} {} [/INST]",
    "target_template": "{}",
    "concat_input_output": True,
    "keep_columns": ["definition", "input", "target", "context_aware_embeds"],
}
 
 
tokenization_kwargs = {
    "tokenizer_name": 'meta-llama/Llama-2-7b-hf',
    "max_input_length": 1024,
    "max_target_length": 1024,
    "padding": "max_length",
    "truncation": True,
    "concat_input_output": True,
    "prefix_tuning": prefix_tuning,
    "n_prefix_tokens": prefix_tokens,
    "decoder_prefix": False,
    "pad_token": 'unk_token'
}

# Datamodule Arguments
datamodule_kwargs = {
    "debug": debug,
    "strategy": strategy,
    "raw_data": raw_data,
    "deduplicate_columns": ["id"],
    "load_from_cache_file": False,
    "num_workers": 12,
    "batch_size": 2,
    "shots": 10000,
    "dev_from_train": -1, ## set to -1 if use dev file for validation, else subsample from train
    "overfit": False,
    "dev_size": 1024,
    "tiny": False,
    "tiny_size": 1024,
    "filter_long_sequences": True,
    "preprocessing_kwargs": preprocessing_kwargs,
    "tokenization_kwargs": tokenization_kwargs,
    "batch_tokenize": True,
    "predict_split": 'dev',

}


## logger arguments
logger_type='wandb'
logger_kwargs = {
    'name': 'llama2/lora/natural_instructions_200k',
    'save_dir':  os.getenv("OUTPUT_DIR", f"{proj_dir}/wandb_logs"),
    'project': os.getenv("WANDB_PROJ_NAME", f"test"),
    'log_model': False,
    'resume':  os.getenv("WANDB_RESUME", "allow"),
}

optimizer_config = {
    "lr": 1e-4,
    "eps": 1e-8,
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingLR",
    
}

lora_config = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules":  ['q_proj','v_proj', 'k_proj', 'lm_head'],
        "task_type": "CAUSAL_LM",
    }


quantization_config = {
    "load_in_8bit":True,
    "bnb_8bit_use_double_quant":True,
    "bnb_8bit_quant_type":"nf8",
    "bnb_8bit_compute_dtype": "bfloat16"
}

generation_kwargs= {
    "max_new_tokens": 30,
    "min_new_tokens": 1,
    "num_return_sequences": 1,  
    "do_sample": False,
    }

# Model Arguments
module_kwargs = {
    "model_name": 'meta-llama/Llama-2-7b-hf',
    "optimizer": 'AdamW',
    "auto_model_class": "AutoModelForCausalLM",
    "prefix_tuning": prefix_tuning,
    "n_prefix_tokens": prefix_tokens,
    "initialize_from_vocab": False,
    
    "optimizer_type": "AdamW",
    "optimizer_config": optimizer_config,
    "gradient_checkpointing": True,
    "quantization_precision": 8,
    "precision": "bf16",
    "tokenization_kwargs": tokenization_kwargs,
    
    "lora": True,
    "lora_config": lora_config,
    "quantization": True,
    "quantization_config": quantization_config,
    
    "generation_kwargs": generation_kwargs,
    
    "freeze_encoder": False,
    "freeze_encoder_layers": [],
    "freeze_decoder": False,
    "freeze_decoder_layers": [],
    "keep_in_fp32_modules": [],
    "resume_from_checkpoint": resume_from_checkpoint,
    "postproc_fn": "identity",
}


# Callbacks
checkpoint_callback=True
checkpoint_callback_kwargs = {
    "dirpath": output_dir,
    "verbose": True,
    "monitor": "val_loss",
    "mode": "min",
    "save_last": True,
    "save_top_k": 1,
    "every_n_train_steps": 10,
    "save_on_train_epoch_end": False
}

# Trainer Arguments
accelerator='auto'
devices="auto"
num_nodes=1
precision="bf16-mixed"
fast_dev_run=False
max_epochs=1
min_epochs=None
max_steps=100000
min_steps=1000
max_time=None
limit_train_batches=None
limit_val_batches=None
limit_test_batches=None
limit_predict_batches=None
overfit_batches=0.0
val_check_interval=.1
check_val_every_n_epoch=1
num_sanity_val_steps=0
log_every_n_steps=50
enable_progress_bar=True
enable_model_summary=True
accumulate_grad_batches=4
gradient_clip_val=0.3
gradient_clip_algorithm='norm'
deterministic=None
benchmark=None
inference_mode=True
profiler=None
detect_anomaly=False
barebones=False
sync_batchnorm=strategy in ['ddp', 'fsdp','fsdp_native', 'ddp_find_unused_parameters_true']
reload_dataloaders_every_n_epochs=0