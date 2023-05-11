from transformers import AutoTokenizer
from datasets import load_dataset
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
import argparse
from mmengine.config import DictAction, Config
from dotenv import load_dotenv
import os

import sys
sys.path.append(os.getcwd())
sys.path.append('/nlp/data/artemisp/apex')

from src.data.pl_dataloaders import *
from pl_modules import *

# Load the variables from the .env file
load_dotenv(os.getcwd()+'/.env')

################################
########## CONFIG ################
################################
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=f'{os.getcwd()}/src/configs/base.py', help='path to config file')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--cfg-options',
                    nargs='+',
                    action=DictAction,
                    metavar="KEY=VALUE",
                    help='overwrite parameters in cfg from commandline')
parser.add_argument("--resume_from_checkpoint", type=str, default=None,help="Path to a folder containing a model checkpoint. Defaults to None")
parser.add_argument("--output_dir", type=str, default=None, help = "Path to save the output. Defaults to None.")

args = parser.parse_args()


cfg = Config.fromfile(args.cfg)
if args.cfg_options:
    cfg.merge_from_dict(args.cfg_options)
    
print(cfg)

#####################################
########## SETUP ################
###################################

if '16' in cfg.precision:
    torch.set_float32_matmul_precision('medium')

seed_everything(cfg.seed, workers=True)

os.environ["TOKENIZERS_PARALLELISM"] = "true" if cfg.num_workers > 0 else "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["TRANSFORMERS_CACHE"]  = os.getenv('CACHE_DIR')

################################
########## WANDB ################
################################
if cfg.report_to == 'wandb':
    os.environ["WANDB_PROJECT"] = os.getenv('WANDB_PROJ_NAME')
    os.environ["WANDB_CACHE_DIR"] = os.getenv('WANDB_DIR')
    os.environ["WANDB_RESUME"] = os.getenv('WANDB_RESUME')
    logger = WandbLogger(name=cfg.run_name, id=cfg.run_name, config=cfg)
elif cfg.report_to == None:
    logger = None
else:
    raise NotImplementedError("More logging options will come soon!..")



################################
########## DATA ################
################################
assert cfg.dataset_name or (cfg.train_file and cfg.dev_file), "Provide a dataset for training in the config!"
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
if cfg.dataset_name:
    dataset = load_dataset(cfg.dataset_name)
else:
    dataset = load_dataset('csv', data_files={"train":cfg.train_file, "validation":cfg.dev_file}, delimiter=',')
if cfg.column_renaming:
    for old_c, new_c in cfg.column_renaming.items():
        dataset = dataset.rename_column(old_c, new_c)
        
if cfg.debug:
    print(dataset)

###########################################
########## MODEL  #########################
############################################
if args.resume_from_checkpoint:
    model = QAModel(cfg=cfg)#load_from_checkpoint(args.resume_from_checkpoint, cfg=cfg)
    data_module = QADataModule(dataset,tokenizer, cfg, batch_size=model.batch_size)
else:
    ################################
    ########## Auto ################
    ################################
    if 'auto' in [cfg.batch_size, cfg.learning_rate]:
        trainer = Trainer(
        # accelerator=cfg.accelerator,
        # strategy=cfg.strategy,
        # devices=cfg.devices,
        num_nodes=cfg.num_nodes,
        precision=cfg.precision,
        fast_dev_run=cfg.fast_dev_run,
        max_epochs=1,
        min_epochs=1,
        limit_train_batches=3,
        limit_val_batches=3,
        limit_test_batches=0,
        limit_predict_batches=0,
        check_val_every_n_epoch=None,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        gradient_clip_val=cfg.gradient_clip_val,
        gradient_clip_algorithm=cfg.gradient_clip_algorithm,
        deterministic=cfg.deterministic,
        benchmark=cfg.benchmark,
        inference_mode=cfg.inference_mode,
        use_distributed_sampler=cfg.use_distributed_sampler,
        profiler=cfg.profiler,
        detect_anomaly=cfg.detect_anomaly,
        barebones=cfg.barebones,
        plugins=None,
        sync_batchnorm=cfg.sync_batchnorm,
        reload_dataloaders_every_n_epochs=cfg.reload_dataloaders_every_n_epochs,
        default_root_dir=f'{args.output_dir}_auto',
        logger=None,
        )
        tuner = Tuner(trainer)

        model = QAModel(cfg, learning_rate=0.0001)   
    if cfg.batch_size == 'auto':
        data_module = QADataModule(dataset, tokenizer, cfg, batch_size=8)
        data_module.setup() 
        optimal_batch_size = tuner.scale_batch_size(model, datamodule=data_module, max_trials=3, init_val=8)
        cfg.batch_size = optimal_batch_size
    else:
        data_module = QADataModule(dataset,tokenizer, cfg, batch_size=cfg.batch_size)
        
    data_module.setup() 
    if cfg.learning_rate == 'auto':
        optimal_lr = tuner.lr_find(model, datamodule=data_module)
        cfg.learning_rate = optimal_lr
    else:
        model = QAModel(cfg)
    

################################
########## Train ################
################################
custom_callback = CustomEvalCallback()
callbacks = [custom_callback]
if cfg.save_top_k > 0:
    checkpoint_callback = ModelCheckpoint(
    dirpath=args.output_dir,
    save_top_k=cfg.save_top_k, 
    verbose=True, 
    monitor=cfg.monitor, 
    mode=cfg.mode
    )
    callbacks.append(checkpoint_callback)

trainer = Trainer(
    accelerator=cfg.accelerator,
    strategy=cfg.strategy,
    devices=cfg.devices,
    num_nodes=cfg.num_nodes,
    precision=cfg.precision,
    fast_dev_run=cfg.fast_dev_run,
    max_epochs=cfg.max_epochs,
    min_epochs=cfg.min_epochs,
    max_steps=cfg.max_steps,
    min_steps=cfg.min_steps,
    max_time=cfg.max_time,
    limit_train_batches=cfg.limit_train_batches,
    limit_val_batches=cfg.limit_val_batches,
    limit_test_batches=cfg.limit_test_batches,
    limit_predict_batches=cfg.limit_predict_batches,
    overfit_batches=cfg.overfit_batches,
    val_check_interval=cfg.val_check_interval if not cfg.tiny else 1,
    check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    num_sanity_val_steps=cfg.num_sanity_val_steps,
    log_every_n_steps=cfg.log_every_n_steps if not cfg.tiny else 1,
    enable_checkpointing=True if cfg.save_top_k > 0 else False,
    enable_progress_bar=cfg.enable_progress_bar,
    enable_model_summary=cfg.enable_model_summary,
    accumulate_grad_batches=cfg.accumulate_grad_batches,
    gradient_clip_val=cfg.gradient_clip_val,
    gradient_clip_algorithm=cfg.gradient_clip_algorithm,
    deterministic=cfg.deterministic,
    benchmark=cfg.benchmark,
    inference_mode=cfg.inference_mode,
    use_distributed_sampler=cfg.use_distributed_sampler,
    profiler=cfg.profiler,
    detect_anomaly=cfg.detect_anomaly,
    barebones=cfg.barebones,
    plugins=None,
    sync_batchnorm=cfg.sync_batchnorm,
    reload_dataloaders_every_n_epochs=cfg.reload_dataloaders_every_n_epochs if not cfg.tiny else cfg.max_epochs,
    default_root_dir=args.output_dir,
    logger=logger,
    callbacks=callbacks,
    )
trainer.fit(model, data_module)
