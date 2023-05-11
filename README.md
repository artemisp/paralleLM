<a name="summary"></a>
# Distributed Training Templates for NLP Tasks
This repo hosts distributed training templates for NLP tasks (currently only has a `Seq2Seq` open and closed book QA example, but soon more support will be available). 

The codebase is based on top of [`PyTorch Lightning`](https://lightning.ai/docs/pytorch/latest/) and [`Huggingface Transformers`](https://huggingface.co/docs/transformers/index). Why you may ask, don't you just use the [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) API of `transformers` for multigpu training? I have found in my experience that it provides little control, and requires a lot of patches that have not been filled. [`Lightning`](https://www.google.com/search?client=safari&rls=en&q=pytorch+lightning&ie=UTF-8&oe=UTF-8) is less automatic, especially for language tasks, but provides more control and robustness in my honest opinion. But, not to worry! You can use tokenizers, models, and datasets from `huggingface` simply by changing the config! :) 

Templates are developed to be compatible with [`balance-my-slurm`](https://github.com/artemisp/balance-my-slurm/tree/main) so check it out! 

Here are some examples of runtimes on T5 models per epoch given 2 RTX A6000 GPUS and `bf16-mixed` precision on 100k examples for Open Book QA (input 512 tokens) and Close Book Q (input 128 tokens) on the MRQA data of Natural Questions  (~120k examples). Validation is run for ~5k examples every epoch. Note that this can vary based on different factors about the cluster status, but it is a good ballpark. 

| model    | task    | iterations | batch size/gpu | time/epoch |
|----------|---------|------------|----------------|------------|
| t5-small | openQA  | 471        | 256            | 6 mins     |
| t5-small | closeQA | 471        | 256            | 4 mins     |
| t5-base  | openQA  | 602        | 200            | 19 mins    |
| t5-base  | closeQA | 602        | 200            | 14 mins    |
| t5-large | openQA  | 12360      | 16             | 4.5 hours  |
| t5-large | closeQA | 12360      | 16             | 3.2 hours  |

You would have to download my preprocessed data to replicate. You can do so as follows:
```
>> mkdir mrqa
>> wget https://faithfulness-data.s3.amazonaws.com/datasets/nq/mrqa/processed/preprocessed_train.csv -O mrqa/preprocessed_train.csv 
>> wget https://faithfulness-data.s3.amazonaws.com/datasets/nq/mrqa/processed/preprocessed_dev.csv -O mrqa/preprocessed_dev.csv 
```
An example command that achieves that for `t5-small` is
```
srun --gpus-per-node=2  --ntasks-per-node=2 --cpus-per-gpu 4 --mem 64GB --constraint 48GBgpu python src/pl_ft.py --cfg src/configs/base.py --output_dir output/ --cfg-options devices=4 min_epochs=1 max_epochs=1 num_workers=4 batch_size=256 use_context=False model_name='t5-small' run_name='t5-small-close-book' dataset_name=None column_renaming=None max_epochs=1 min_epochs=1 learning_rate=0.0001
```

<a name="toc"></a>
## Table of Contents
1. [Environment Variables](#environment-variables-section)
2. [Installation](#installation)
3. [Files and Skeleton](#skeleton)
4. [Configuration](#config)
5. [Tips and Tricks](#tips)

<a name="environment-variables-section"></a>
## Environment Variables
Update `.env` file
* Set `DATA_DIR` which is the directory where you will download the relevant data. 
* Set `WANDB_PROJ_NAME` which is the name of the project in wandb.
* Set `WANDB_DIR` which is the name of the directory where `wandb` stores its data
* Set `WANDB_RESUME` (see [documentation](https://docs.wandb.ai/guides/track/environment-variables#optional-environment-variables)) which determines whether wandb runs resume in the same panels.  
* Set `CACHE_DIR` the cache dir for `transformers` and other caching purposes. 
* Set `CHECKPOINT_DIR` the checkpoint directory. 
* Set `PREDICTION_DIR` the predictions directory. 

<a name="installation"></a>
## Installation
Setting up an environment for multigpu training is no joke! Things break easily if not executed in the correct order. Here you can find step by step instructions on how to set up the environment. 

Here we assume `CUDA` version 11.7 which are compatible with our lab clusters. Instructions for other CUDA versions are soon to come in Section [Other CUDA Versions](#other-cuda-versions) below. 

```
>> conda create -n test_me python=3.10
>> conda activate test_me
>> git clone https://github.com/NVIDIA/apex
>> cd apex
```
and run the following **from inside an interactive gpu instance**:
```
>> conda activate test_me
>> python -m pip install packaging
>> conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
>> python -m pip install -r requirements.txt
>> python -m pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
>> conda install lightning -c conda-forge
```
Now... `apex` takes some time to install. You can speed it up by using [`Ninja`](https://ninja-build.org) to make the compilation faster, but.. hey! what better excuse to go on and play some video games, go for a run, or simply nap than `my code is running...`.

`lightning` also takes some time to install because `conda` has to resolve a lot of conflicts. If you want to use a faster (like a LOT FASTER) package manager built on top of conda (i.e. same commands), I highly recomend [mamba](https://github.com/mamba-org/mamba)! 

#### Troubleshooting
* Sometimes you may get into issues with importing `PyTorch`. If so uninstall it, from *both conda and pip* and reinstall with the `--force-reinstall` flag. 
* Make sure your installation is from inside a gpu interractive instance from the type of nodes you will be using for your experiments.
* If all fails, will need to start again with a clean build. 

<a name="other-cuda"></a>
#### Other CUDA Versions
`WILL BE UPDATED SOON!` But here are some important links:
* Find the correct `PyTorch` installation for your system requirements [here](https://pytorch.org/get-started/locally/)
* Make sure you have the system requirements for [Apex](https://github.com/NVIDIA/apex) and make sure you install it from source since it requires `cpp` compiling. 


<a name="skeleton"></a>
## Files and Skeleton
```
├── src
│   ├── configs
│   │   ├── base.py 
│   │   ├── <your_own_config.py>
│   ├── data
│   │   ├── pl_dataloaders.py # lightning data module
│   │   ├── postprocessing.py # postprocessing functions
│   │   ├── preprocessing.py # preprocessing functions
│   ├── predict
│   │   ├── predict.py # predict on a test csv file!
│   │   ├── qa_utils.py # borrowed from official T5 repo
│   │   ├── utils.py # generation/decoding functoins
├── pl_ft.py # main file for finetuning
├── pl_modules.py # custom lighnting modules for models
├── .env # environment variables using pydotenv
├── requirements.txt # some dependencies
└── .gitignore # keeps your .git clean and neat
```
Now let's look at each of them in turn:

<a name="data"></a>
### Data
This module handles dataloading, preprocessing, and post-processing. 

#### `preprocessing.py`

Contains a function `get_inputs_and_targets` that augments a [`datasets.Dataset`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset) with a inputs and labels that can be directly fed to a tokenizer. The function is called in `data.pl_dataloaders.QADataset` and in `predict.py` to convert the test csv to a tokenized dataset. 


This is the example below for [MRQA](https://github.com/mrqa/MRQA-Shared-Task-2019) data
```
    Input dataset form:
    {
        'id': <mrqa_example_id>
        'context': <example context>
        'question': <example question>
        'answer': <sample_answer>
        'split': <train/dev>
    }
    updates with entries
    {
        'input': 'question: <question> context: <context>'
        'target': 'answer: <answer>'
    }
    
    if context is set to false:
     {
        'input': 'question: <question>'
        'target': 'answer: <answer>'
    }
    
    Args:
        dataset (dataset.Dataset): A preprocessed dataset in the form of `dataset.Dataset` to process.
        use_context (bool, optional): A boolean indicating whether to use context or not. Defaults to `True`.
        num_proc (int, optional): An integer indicating the number of processes to use for mapping the dataset. Defaults to `1`.
        load_from_cache_file (bool, optional): A boolean indicating whether to load the dataset from a cache file. Defaults to `False`.
        column_dict (dict, optional): A dictionary indicating the column names for the context, question, and answer. The keys for the dictionary are `context`, `question`, and `answer`. Defaults to `{"context":"context", "question":"question", "answer": "answer"}`.

    Returns:
        A dataset.Dataset: An augmented `dataset.Dataset` with the following format for each example:
        {
            'id': <mrqa_example_id>
            'context': <example context>
            'question': <example question>
            'answer': <sample_answer>
            'split': <train/dev>
            'input': 'question: <question> context: <context>' if use_context is True else 'question: <question>'
            'target': 'answer: <answer>'
        }
```

#### `postprocessing.py`

Here you include any string processing functions that you want to apply on the predictions of your model before it is passed for evaluation. More specifically, they can be called in `predict.py` and `pl_modules.CustomEvalCallback`.

#### `pl_dataloaders.py`

Here we have the dataloading pyorch lightning modules. The main classes are the [`QADataset(torch.utils.data.Dataset)`](https://pytorch.org/docs/stable/data.html#dataset-types) and the [`QADataModule(pl.LightningDataModule)`](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) which is passed in the lightning [`Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html) to handle dataloading and tokenization. 

How to initialize each of these modules you ask.. simple! 
```
"""
Args:
    dataset (Dataset): A preprocessed dataset in the form of `datasets.Dataset` to process.
    tokenizer: The tokenizer object to use for tokenization.
    cfg: The configuration object that contains the required parameters for processing the dataset.
"""
from datasets import load_dataset
from tokenizers import AutoTokenizer
dataset = load_dataset('squad')
tokenizer = AutoTokenizer.from_pretrained('t5-small')
cfg = 'src/configs/base.py'
qa_dataset = QADataset(dataset,tokenizer,cfg)
```
And the data module can be loaded as follows
```
from datasets import load_dataset
from tokenizers import AutoTokenizer
dataset = load_dataset('squad')
tokenizer = AutoTokenizer.from_pretrained('t5-small')
cfg = 'src/configs/base.py'
qa_dataset = QADataset(dataset,tokenizer,cfg)
qa_data_module = QADataModule(dataset, tokenizer, cfg)
```

<a name="models"></a>
### `pl_modules.py`

The most important method of this file is the [`QAModel(pl.LightningModule)`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html). This PyTorch Lightning module encapsulates a Hugging Face transformer model for Seq2Seq tasks and is optimized by the `pl.Trainer` in `pl_tf.py`. 

A little bit more details can be found in the docstring
```
        Attributes
        ----------
        model : transformers.AutoModelForSeq2SeqLM
            The Seq2Seq transformer model.
        tokenizer : transformers.AutoTokenizer
            The tokenizer associated with the transformer model.
        learning_rate : float
            The learning rate for the optimizer.
        cfg : object
            The configuration object with model and training parameters.
        batch_size : int
            The batch size for the training.

        Methods
        -------
        forward(input_ids, attention_mask, labels=None):
            Performs a forward pass through the model.
        training_step(batch, batch_idx):
            Defines the procedure during one training step.
        validation_step(batch, batch_idx):
            Defines the procedure during one validation step.
        test_step(batch, batch_idx):
            Defines the procedure during one testing step.
        configure_optimizers():
            Configures the optimizer for training.
        """
```

Here is how to initialize it: 

```
cfg = 'src/configs/base.py'
model = QAModel(cfg)
```
and if you want to load from a particular checkpoint `ckpt` you can do the following:
```
checkpoint = checkpoint.ckpt
cfg = 'src/configs/base.py'
model = QAModel.load_from_checkpoint(checkpoint,cfg=cfg)
```
That was simple haha :P

<a name="main-ft"></a>
### `pl_ft.py`
Now this is the main function that trains our model. Most of its arguments are specified in the config file. For more details see Section [Conifguration](#config) below. 

The file handles a series of different things simply by changing the in the config: This is a script for training a transformer model using PyTorch Lightning. It includes several steps: configuration setup, data loading, model creation, logging, and training.

If you want to run this in interractive mode, maybe to debug some of your own additions (btw pull requests are welcome!) you can do something along the following lines:
```
>> conda activate test_me
>> srun --gpus 1 --ntasks-per-node=1 --mem 64GB -w nlpgpu04 python src/pl_ft.py --cfg src/configs/base.py --output_dir output/ --cfg-options devices=1
```
This will run on a single gpu. Now if you want to distribute to more you can run 
```
srun --gpus 2 --ntasks-per-node=2 --mem 64GB -w nlpgpu04 python src/pl_ft.py --cfg src/configs/base.py --output_dir output/ --cfg-options devices=2
``` 

For debugging purposes you can set the `tiny`, `tiny_size`, `debug` and/or `overfit` arguments in the config as follows:
```
srun --gpus 1 --ntasks-per-node=1 --mem 64GB -w nlpgpu04 python src/pl_ft.py --cfg src/configs/base.py --output_dir output/ --cfg-options devices=2 tiny=true tiny_size=4 overfit=true debug=true
``` 

With this configuration and a `batch_size=128` and mixed precision,  I can run `t5-base` on the [`Natural Questions`]() subset of MRQA for closed book (~100k examples) in under 10 minutes per epoch! 


The module can take the following arguments, and they are compatible with [`balance-my-slurm`](https://github.com/artemisp/balance-my-slurm/tree/main) so head on there to copy the sbatch files :) All you need to do is change the `SLURM_JOB_SCRIPT` to `pl_ft.py` in line 10 of [`continuous_deployment.sh`](https://github.com/artemisp/balance-my-slurm/blob/main/continuous_deployment.sh)


- `--cfg`: This argument specifies the path to the configuration file. It defaults to the `base.py` file in the `configs` directory located in the current working directory. The configuration file contains settings for the model and the training process.

- `--local_rank`: This is an integer argument that defaults to 0. In the context of distributed computing, this would be the rank of the process on the local node.

- `--cfg-options`: This is an optional argument that allows you to overwrite parameters in the configuration file from the command line. The values must be given in the `KEY=VALUE` format. You can specify multiple `KEY=VALUE` pairs by separating them with spaces.

- `--resume_from_checkpoint`: This argument allows you to continue training from a previously saved checkpoint. You need to provide the path to the folder containing the model checkpoint. If this argument is not provided, the training will start from scratch.

- `--output_dir`: This argument specifies the directory where the output (e.g., model checkpoints, training logs) should be saved. If not provided, no output will be saved.


<a name="predict"></a>
### `predict`

This module predicts and saves the predictions under `PREDICTION_DIR` from `.env` and computes some basic metrics. 

Here is an example of how to run the predict module if you do not want to use a checkpoint and simply want to predict on the validation set of `sciq` which is specified in the `cfg` for open book qa. 
```
srun --gpus 1 --ntasks-per-node=1 --mem 64GB -w nlpgpu04 python src/predict/predict.py --cfg src/configs/base.py --input_to_column_dict '{"question": "question", "context": "context"}' --target_to_column_dict '{"answer": "answer"}' --pred_column prediction --num_workers 1 --cfg-options 
```
and here is an example with a checkpoint and a custom csv (not included) for closed book qa.

```
srun --gpus 1 --ntasks-per-node=1 --mem 64GB -w nlpgpu04 python src/predict/t5/predict.py --cfg src/configs/base.py --use_checkpoint output/debug/epoch=7-step=6011.ckpt --predict_on_train 1 --predict_csv_files /mrqa/preprocessed_dev.csv --input_to_column_dict '{"question": "question", "context": "context"}' --target_to_column_dict '{"answer": "answer"}' --pred_column parametric_prediction --num_workers 1 --cfg-options use_context=false
```
Here is a brief description of each argument for `predict.py`:

- `--cfg`: This argument is used to specify the path to the configuration file. The default value is `f'{os.getcwd()}/src/configs/t5/configs/base.py'` which means that the configuration file is located at 'base.py' under the directory `src/configs/t5/configs/` relative to the current working directory.

- `--use_checkpoint`: This argument is used to specify a specific checkpoint for the model. If this argument is not provided or is empty, the model is initialized from the pre-trained model specified in `cfg.model_name`.

- `--predict_on_train`: This argument is a flag to determine whether to make predictions on the training set. The default value is 1, meaning predictions will be made on the training set. If set to 0, predictions will not be made on the training set.

- `--predict_csv_files`: This argument is used to provide one or more CSV files on which to make predictions. The argument takes a list of file paths.

- `--input_to_column_dict`: This argument is a JSON string that maps the 'context' and 'question' to their corresponding columns in the CSV file. If 'context' is empty, it is not used. The default mapping is `{"question": "question", "context": "context"}`.

- `--target_to_column_dict`: This argument is a JSON string that maps 'answer' to its corresponding column in the CSV file. The default mapping is `{"answer": "answer"}`.

- `--pred_column`: This argument is used to specify the name of the column where predictions will be stored in the CSV file. The default column name is 'prediction'.

- `--cfg-options`: This argument is used to overwrite parameters in the configuration file from the command line. The argument takes a list of key-value pairs.

- `--num_workers`: This argument is used to specify the number of CPUs to be used for data loading. The default value is 1.

<a name="config"></a>
## Configuration

Okay.. all good till now, but of course you want to take some control. Don't you worry! A lot of the things you would want to do can simply be achieved by changing a single variable in the config! 

Note that you can inherit the base config and just change the parameters you care about by including your `config.py` in the same directory and on the first line including:
```
_base_ = 'base.py'
```
and you can inherit from your own config, and so on! Simply by specifying the `_base_`.


Here is a summary of what each parameter does and some possible alternatives.  
#### Reproducibility
```
seed=42
```
We use [`pl.seed_everything(cfg.seed, workers=True)`](https://pytorch-lightning.readthedocs.io/en/1.6.3/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything) to make sure that the experiments can be reproduced. It sets the random seeds for `numpy`, `pytorch`, and `random` globally. 

#### Data
```
dataset_name = 'sciq' # huggingface dataset name
column_renaming = {"support":"context", "correct_answer": "answer"}
train_file = None
dev_file = None
dev_from_train=.05 ## set to -1 if use dev file for validation, else subsample from train
filter_long_sequences=True
```
1. `dataset_name`: Name of the dataset being used (from [Huggingface Hub](https://www.google.com/search?client=safari&rls=en&q=huggingface+hub&ie=UTF-8&oe=UTF-8))
2. `column_renaming`: Renames specific columns in the dataset.
3. `train_file` and `dev_file`: Paths or names of the training and development dataset files.
4. `dev_from_train`: Determines the proportion of the training dataset used for validation.
5. `filter_long_sequences`: Controls whether long sequences are filtered from the dataset.


#### Debug

```
debug=False
tiny=False
tiny_size=1024
overfit=False
```

1. `debug=False`: Turns on/off debugging mode, indicating that the code will not provide additional debugging information or behavior.
2. `tiny=False`: Set to `True` to use a small portion of the data specified by `tiny_size` for debugging. 
3. `tiny_size=1024`: Specifies the size (in this case, 1024) for the "tiny" configuration. 
4. `overfit=False`: If `True` sets validation set to be the same as train. 

#### Task Type
```
use_context=False
shots=-1
```
1. `use_context=False`: Specifies whether the code uses context information as input.

2. `shots=-1`: Specifies how many shots to train with, if set to `-1` trains on full dataset.  

#### Logging Arguments
```
run_name = "t5-small-open-book"
report_to='wandb'
```
1. `run_name`: the name of the experiment, used for logging and output folders.
2. `report_to`: currently only supports [`wandb`](https://wandb.ai) but support for [`tensorboard`](https://www.tensorflow.org/tensorboard/get_started#:~:text=TensorBoard%20is%20a%20tool%20for,dimensional%20space%2C%20and%20much%20more.) will be provided. If set to `None` no reporting occurs.

#### Model and Tokenizer
```
model_name = 't5-small'
input_max_length = 256
output_max_length = 32
num_workers=1
learning_rate='auto'
batch_size='auto'
```
1. `model_name`: Specifies the name of the model being trained - if no checkpoint is provided, this is the argument that is passed in [`transformers.AutoModel.from_pretrained`](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html) and [`transformers.AutoTokenizer.from_pretrained`](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#autotokenizer)  

2. `input_max_length`: input max length passed to `tokenizer`

3. `output_max_length`: output max length for decoding

4. `num_workers`: Specifies the number of workers for parallel processing.

5. `learning_rate`: model learning rate, if set to `auto` uses [`pytorch_lightning.Tuner`](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.tuner.tuning.Tuner.html) to find the ideal one. `Tuner` can only be used in non-parallel. 

6. `batch_size`: model batch_size, if set to `auto` uses [`pytorch_lightning.Tuner`](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.tuner.tuning.Tuner.html) to find one that maximizes gpu usage.`Tuner` can only be used in non-parallel, but once you find that optimal batch size you can simply pass it as an argument in the `cfg`.

#### Generation Arguments
```
repetition_penalty=2.5
length_penalty=1.0
no_repeat_ngram_size=3
num_beams=1
early_stopping=True
```

Defaults for `T5` from [here](https://github.com/ellaneeman/disent_qa/blob/main/config.json). These are the arguments used for `tokenizer.generate` same function as the [`Generation Configs`](https://huggingface.co/docs/transformers/main_classes/text_generation)

#### optimizer
```
optimizer = 'Adafactor' 
```
Selects `optimizer` for the model. Currently supported options:
[`transformers.Adafactor`](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor), [`torch.optim.AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW), [`torch.optim.Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam).


#### Trainer Arguments
```
accelerator='auto'
strategy='ddp'
devices=2
num_nodes=1
precision="bf16-mixed"
fast_dev_run=False
max_epochs=1000
min_epochs=50
max_steps=-1
min_steps=None
max_time=None
limit_train_batches=None
limit_val_batches=None
limit_test_batches=None
limit_predict_batches=None
overfit_batches=0.0
val_check_interval=200
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
use_distributed_sampler=False
profiler=None
detect_anomaly=False
barebones=False
sync_batchnorm=True
reload_dataloaders_every_n_epochs=0
```

These are the training arguments passed in the `pytorch_lightning.Trainer`. You can refer to the documentation here for details. However, I want to point your attention to some details specific to our cluster and some information about parallelization. 

##### Mixed Precision
The options for mixed precision are the folllowing - 
- Double precision (`64`), 
- full precision (`32`),
- half precision AMP (`16-mixed`), or 
- bfloat16 precision AMP (`bf16-mixed`).

However, note that you can only use `bf16-mixed` in nlpgpu04-7. 

##### Strategies for Parallelization

See more info [here](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html). I just want to know, that especially if new in parallelization, **opt for** `ddp` *"where the maximum trainable model size and batch size do not change with respect to the number of GPUs, memory-optimized strategies can accommodate bigger models and larger batches as more GPUs are used. This means as you scale up the number of GPUs, you can reach the number of model parameters you’d like to train"*. It is not the most efficient, but will cause the least headaches. 



<a name="tips"></a>
## Tips and Tricks

* You can only use `bf16-mixed` in `nlpgpus04-07`. Otherwise you can use the standard `16-mixed`.
* For some reason, `T5` models lead to `nan` losses with `16-mixed` but not with `bf16-mixed`. I hypothesize it is related to [this](https://github.com/huggingface/transformers/issues/10830) thread.
* Use the `auto` parameters in the config to maximize gpu memory usage! It uses `pl.Tuner` - see more documentation [here](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html). It is important to note that `Tuner` does not work in parallelized mode so run on a single gpu to figure out the optimal batch size and set it on the config for future use :) 
* If you keep getting an error about your strategy of parallelization, even if you change it to something new, then `lightning` is storring your previous selections. You should comment out the arguments below in `pl_ft.py[166:168]` and let your code run with defaults. It should fix the issue. 
```
accelerator=cfg.accelerator,
strategy=cfg.strategy,
devices=cfg.devices,
```
