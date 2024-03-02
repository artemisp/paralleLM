if __name__ == "__main__":
    
    import sys
    import os
    sys.path.append(os.getcwd())
    
    import datasets
    datasets.disable_caching()
    
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from mmengine.config import DictAction, Config
    
    from dotenv import load_dotenv
    
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    import time
    import argparse
    import pprint
    import json

    from src.data.pl_dataloaders import *
    from src.models.pl_modules import *


    start_time = time.time()
    import os
    ################################
    ########## CONFIG ################
    ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=f'{os.getcwd()}/src/configs/train/llama.py', help='path to config file')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        metavar="KEY=VALUE",
                        help='overwrite parameters in cfg from commandline')
    args = parser.parse_args()


    cfg = Config.fromfile(args.cfg)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
        
    pprint.pprint(cfg.to_dict(), indent=4)
    
    
    ################################
    ########## OUTPUT DIR #########
    ################################
    # todo: create new output dir if exists
    output_dir = cfg.get('output_dir', None)
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Output directory ({output_dir}) already exists and is not empty.")
    os.makedirs(output_dir, exist_ok=True)
    json.dump(cfg.to_dict(), open(os.path.join(output_dir, "config.json"), "w"))
    
    
    ################################
    ########## ENVIRONMENT #########
    ################################
    # Load the variables from the .env file
    load_dotenv(os.getcwd()+'/.env')
    # os.environ["TOKENIZERS_PARALLELISM"] = "true" if cfg.num_workers > 0 else "false"
    os.environ["TRANSFORMERS_CACHE"]  = os.getenv('CACHE_DIR', "./.cache")
    
    #####################################
    ########## SETUP ################
    ###################################
    if '16' in str(cfg.get('precision', '32')):
        torch.set_float32_matmul_precision('medium')
    else:
        torch.set_float32_matmul_precision('high') 
    pl.seed_everything(cfg.get('seed', 42), workers=True)
    
    data_module = CustomDataModule(**cfg.get('datamodule_kwargs', {}))
    module = CustomModule(tokenizer=data_module.tokenizer, **cfg.get('module_kwargs', {}), predict=True)

    ################################
    ########## Train ################
    ################################
    trainer = pl.Trainer(
        strategy=cfg.get('strategy', 'auto'),
        default_root_dir=output_dir,
        logger=None,
        callbacks=[],
        devices=cfg.get('devices', 'auto'),
        num_nodes=cfg.get('num_nodes', 1),
        precision=cfg.get('precision', 32),
        fast_dev_run=cfg.get('fast_dev_run', False),
        max_epochs=cfg.get('max_epochs', 3),
        min_epochs=cfg.get('min_epochs', 1),
        max_steps=cfg.get('max_steps', 10000),
        min_steps=cfg.get('min_steps', None),
        max_time=cfg.get('max_time', None),
        use_distributed_sampler=cfg.get('use_distributed_sampler', False),
        limit_train_batches=cfg.get('limit_train_batches', None),
        limit_val_batches=cfg.get('limit_val_batches', None),
        limit_test_batches=cfg.get('limit_test_batches', None),
        limit_predict_batches=cfg.get('limit_predict_batches', None),
        overfit_batches=cfg.get('overfit_batches', None),
        val_check_interval=cfg.get('val_check_interval', None),
        check_val_every_n_epoch=cfg.get('check_val_every_n_epoch', None),
        num_sanity_val_steps=cfg.get('num_sanity_val_steps', 0),
        log_every_n_steps=cfg.get('log_every_n_steps', 50),
        enable_checkpointing=cfg.get('checkpoint_callback', False),
        enable_progress_bar=cfg.get('enable_progress_bar', False),
        enable_model_summary=cfg.get('enable_model_summary', False),
        accumulate_grad_batches=cfg.get('accumulate_grad_batches', 1),
        gradient_clip_val=cfg.get('gradient_clip_val', None),
        gradient_clip_algorithm=cfg.get('gradient_clip_algorithm', "norm"),
        deterministic=cfg.get('deterministic', False),
        benchmark=cfg.get('benchmark', False),
        inference_mode=cfg.get('inference_mode', True),
        profiler=cfg.get('profiler', False),
        detect_anomaly=cfg.get('detect_anomaly', False),
        barebones=cfg.get('barebones', False),
        plugins=None,
        sync_batchnorm=cfg.get('sync_batchnorm', False),
        reload_dataloaders_every_n_epochs=cfg.get('reload_dataloaders_every_n_epochs', 0),
        )
    

    predictions = trainer.predict(module, data_module, return_predictions=True, ckpt_path=cfg.get('resume_from_checkpoint', None))
    json.dump(predictions, open(os.path.join(output_dir, f"predictions_rank{torch.distributed.get_rank()}.json"), "w"))
    torch.distributed.barrier()
    #
    predictions = []
    if torch.distributed.get_rank() == 0:
        for r in range(0, torch.distributed.get_world_size()):
            predictions.extend(json.load(open(os.path.join(output_dir, f"predictions_rank{r}.json"), "r"))) 
        # flatten predictions
        predictions = [r for batch in predictions for r in batch]
        json.dump(predictions, open(os.path.join(output_dir, "predictions.json"), "w"))
        
        import evaluate
        metrics = cfg.get('metrics', [])
        computed_metrics = {}
        for metric in metrics:
            metric_fn = evaluate.load(metric)
            value = metric_fn(references=[r['target'] for r in predictions], predictions=[r['prediction'] for r in predictions])
            computed_metrics[metric] = value
        

        print(computed_metrics)
        json.dump(computed_metrics, open(os.path.join(output_dir, "metrics.json"), "w"))
        end_time = time.time()
        print(f"Total time taken: {end_time-start_time} seconds")
