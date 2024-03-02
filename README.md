<a name="summary"></a>
# 🌍 Distributed Training Templates for NLP Tasks 📚
This repo hosts a distributed training template for NLP tasks. 🚀

The codebase is based on top of [`PyTorch Lightning`](https://lightning.ai/docs/pytorch/latest/) and [`Huggingface Transformers`](https://huggingface.co/docs/transformers/index). Why you may ask, don't you just use the [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) API of `transformers` for multigpu training? I have found in my experience that it provides little control, and requires a lot of patches that have not been filled. [`Lightning`](https://www.google.com/search?client=safari&rls=en&q=pytorch+lightning&ie=UTF-8&oe=UTF-8) is less automatic, especially for language tasks, but provides more control and robustness in my honest opinion. But, not to worry! You can use tokenizers, models, and datasets from `huggingface` simply by changing the config! :) 

Templates are developed to be compatible with [`balance-my-slurm`](https://github.com/artemisp/balance-my-slurm/tree/main) so check it out! 🧐

An example config file to run is provided in `src/configs/train/llama_mrqa.py`. Make sure to download the data to run it
 You can do so as follows:
```
>> mkdir mrqa
>> wget https://faithfulness-data.s3.amazonaws.com/datasets/nq/mrqa/processed/preprocessed_train.csv -O mrqa/preprocessed_train.csv 
>> wget https://faithfulness-data.s3.amazonaws.com/datasets/nq/mrqa/processed/preprocessed_dev.csv -O mrqa/preprocessed_dev.csv 
```
Then you can train the model by:
```
 srun --gpus 1 --nodes 1 --mem-per-cpu 12GB  --constraint 48BGgpu --ntasks-per-node 1 --cpus-per-gpu 10 /nlp/data/artemisp/mambaforge/envs/test_me/bin/python  src/pl_ft.py --cfg /nlp/data/artemisp/multigpu-lm-templates/src/configs/train/llama_mrqa.py
```

<a name="toc"></a>
## 📑 Table of Contents
1. [🌐 Environment Variables](#environment-variables-section)
2. [🔧 Installation](#installation)
3. [📁 Files and Skeleton](#skeleton)
4. [⚙️ Configuration](#config)
5. [➕ Add your own dataset](#add_ds)
6. [🏋️‍♂️ Train](#train)
7. [📊 Evaluate](#evaluate)


<a name="environment-variables-section"></a>
## 🌐 Environment Variables
Update `.env` file
* Set `DATA_DIR` which is the directory where you will download the relevant data. 📂
* Set `WANDB_PROJ_NAME` which is the name of the project in wandb. 🏷️
* Set `WANDB_DIR` which is the name of the directory where `wandb` stores its data 🗂️
* Set `WANDB_RESUME` (see [documentation](https://docs.wandb.ai/guides/track/environment-variables#optional-environment-variables)) which determines whether wandb runs resume in the same panels. 🔄  
* Set `CACHE_DIR` the cache dir for `transformers` and other caching purposes. 💾
* Set `OUTPUT_DIR` the checkpoint and results directory. 🎯
* Set `HF_ACCESS_TOKEN` the huggingface access token for private models. See how to retrieve it [here](https://huggingface.co/docs/hub/en/security-tokens) 🔑

<a name="installation"></a>
## 🔧 Installation
Setting up an environment for multigpu training is no joke! Things break easily if not executed in the correct order. Here you can find step by step instructions on how to set up the environment. 🛠️

In installing `PyTorch` we assume `CUDA` version 12.0 are compatible with our lab clusters. For other versions see the installation [page](https://pytorch.org/get-started/locally/). 🔥

```
>> conda create -n test_me python=3.10
>> conda activate test_me
>> conda activate test_me
>> conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
>> python -m pip install -r requirements.txt
```

If you want to use a faster (like a LOT FASTER) package manager built on top

 of conda (i.e. same commands), I highly recommend [mamba](https://github.com/mamba-org/mamba)! 🐍


<a name="skeleton"></a>
## 📁 Files and Skeleton
```
├── src
│   ├── common
│   │   ├── checkpoint_utils.py # utility functions for checkpointing
│   ├── configs
│   │   ├── base_llama.py # example config to LoRA [LLaMA2-7b](https://huggingface.co/meta-llama/Llama-2-7b) on ]Natural Instructions](https://huggingface.co/datasets/Muennighoff/natural-instructions) from HF Datasets.
│   │   ├── train
│   │   ├   ├── llama_mrqa.py # example to LoRA [LLaMA2-7b](https://huggingface.co/meta-llama/Llama-2-7b) on [MRQA](https://huggingface.co/datasets/mrqa) from local csv.
│   │   ├── eval
│   ├── data
│   │   ├── pl_dataloaders.py # lightning data module
│   │   ├── postprocessing.py # postprocessing functions
│   │   ├── preprocessing.py # preprocessing functions
│   │   ├── data_utils.py # utility string processing functions
│   ├── models
│   │   ├── pl_modules.py # custom lighnting modules for models
│   │   ├── soft_embedding.py # implements [prefix tuning](https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py)
│   ├── pl_predict.py # main file for predictions
│   ├── pl_ft.py # main file for finetuning
├── .env # environment variables using pydotenv
├── requirements.txt # some dependencies
└── .gitignore # keeps your .git clean and neat
```
Now let's look at each of them in turn:

<a name="data"></a>
### 📊 Data: The Heart of Your NLP Adventure! 🚀
This module handles dataloading, preprocessing, and post-processing. 

#### `data/pl_dataloaders.py`

* `CustomDataset`: A subclass of `torch.utils.data.Dataset`, designed for flexible data handling. It supports initialization with datasets in various formats, optional tokenizer integration, and custom preprocessing. It is used by `CustomDataModule`.
* `CustomDataModule`: Extends `pl.LightningDataModule` to organize data loading, preprocessing, and setup for different phases like training and validation. It supports distributed training and custom tokenization and preprocessing workflows.
 """
        A PyTorch Lightning DataModule for storing and managing custom data.
        
        Some kwargs that can be set are:
        - `raw_data` (str, datasets.Dataset, dict): The raw data to process.
        - `preprocessing_kwargs` (dict): The preprocessing arguments to use for processing the raw data. More specifically,
            - `max_length` (int): The maximum length of the input sequence.
            - `max_target_length` (int): The maximum length of the target sequence.
            - `num_workers` (int): The number of workers to use for processing the data.
            - `load_from_cache_file` (bool): Whether to load the data from the cache file.
            - `batch_tokenize` (bool): Whether to batch tokenize the data, if set to False the entire dataset is preloaded in memory
        - `tokenization_kwargs` (dict): The tokenization arguments to use for tokenizing the data. More specifically,
            - `padding` (str): The padding strategy to use for the sequences.
            - `truncation` (str): The truncation strategy to use for the sequences.
            - `max_length` (int): The maximum length of the input sequence.
            - `max_target_length` (int): The maximum length of the target sequence.
            - `num_workers` (int): The number of workers to use for tokenization.
            - `load_from_cache_file` (bool): Whether to load the data from the cache file.
        - `strategy` (str): The strategy to use for distributed processing.
        - `splits` (list): The splits to use for processing the data.
        - `dev_size` (float, int): The size of the development set.
        - `tiny` (bool): Whether to use a tiny dataset for debugging.
        - `tiny_size` (int): The size of the tiny dataset.
        - `overfit` (bool): Whether to overfit the model, i.e. use the training set as the validation set.
        - `shots` (int): The number of shots to use for training.

        Args:
            dataset (Dataset): A preprocessed dataset in the form of `datasets.Dataset` to process.
            tokenizer: The tokenizer object to use for tokenization or str with tokenizer name.
            batch_size (int, optional): The batch size to use for training and inference. Defaults to `None`.
        """

#### `data/postprocessing.py`
Define postprocessing functions here. They are accesed in `models.pl_module.CustomModule` prediction step. They can be defined in `datamodule_kwargs` in the config by their name `datamodule_kwargs: {"postproc_fn": <func_name>}`. Each function in postprocessing accepts a single string. 

#### `data/preprocessing.py`
Define preprocessing functions here. It is used by `data.pl_dataloaders.CustomDataModule` for template formatting, and tokenization. The relevant arguments in the config are `preprocessing_kwargs` and `tokenization_kwargs`.

<a name="data"></a>
### Models 🤖


#### `models/pl_modules.py`
*  `CustomModule`: A `lightning` wrapper around a `transformers` model that allows for training in `LoRA` or prefix tuning mode using the implementation from [here](https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py) as well as quantization. It allows for distributed training, and high control of the training processes. The `training_step` method can be adapted for different loss functions if necessary. 
 """
        A PyTorch Lightning module that encapsulates a Hugging Face transformer model
        for Seq2Seq tasks like machine translation or text summarization.

        Attributes
        ----------
        learning_rate : float
            The learning rate for the optimizer. If None, it will be taken from the optimizer_config.
        tokenizer : transformers.AutoTokenizer
            The tokenizer to use for the model. If None, it will be taken from the tokenization_kwargs.
        predict : bool
            Whether the model is used for prediction or not. If True, the predict_step method will be used.
        kwargs : dict
            A dictionary with the configuration for the model, the optimizer, the tokenizer, etc.
            Possible arguments are:
            - debug : bool
                Whether to print debug information or not.
            - optimizer_type : str
                The type of optimizer to use. Possible values are 'Adam', 'AdamW', and 'Adafactor'.
            - optimizer_config : dict
                A dictionary with the configuration for the optimizer.
            - generation_kwargs : dict
                A dictionary with the configuration for the generation method of the model.
            - auto_model_class : str
                The name of the class of the model to use from transformers. Possible values are 'AutoModelForSeq2SeqLM' and 'AutoModelForCausalLM'.
            - lora : bool
                Whether to use LORA or not.
            - lora_config : dict
                A dictionary with the configuration for LORA.
            - prefix_tuning : bool
                Whether to use prefix tuning or not.
            - n_prefix_tokens : int
                The number of tokens to use for prefix tuning.
            - initialize_from_vocab : bool
                Whether to initialize the prefix from the vocabulary or not.
            - quantization : bool
                Whether to use quantization or not.
            - quantization_config : dict
                A dictionary with the configuration for quantization.
            - quantization_precision : str
                The precision to use for quantization. Possible values are '4', '8', and '16'.
            - keep_in_fp32_modules : list
                A list with the names of the modules to keep in FP32.
            - freeze_encoder : bool
                Whether to freeze the encoder or not.
            - freeze_encoder_layers : list
                A list with the indices of the layers of the encoder to freeze.
            - freeze_decoder : bool
                Whether to freeze the decoder or not.
            - freeze_decoder_layers : list
                A list with the indices of the layers of the decoder to freeze.
            - postproc_fn : str
                The name of the function to use for postprocessing the predictions.
            - gradient_checkpointing : bool
                Whether to use gradient checkpointing or not.
            - model_name : str
                The name of the model to use from Hugging Face.
            - tokenization_kwargs : dict
                A dictionary with the configuration for the tokenizer.
            
            Methods
            -------
            setup(stage)
                Initializes the model, the tokenizer, and the optimizer.   
            forward(input_ids, attention_mask, labels=None, output_hidden_states=False, output_attentions=False)
                Performs a forward pass of the model.
            training_step(batch, batch_idx)
                Performs a training step of the model.
            predict_step(batch, batch_idx)
                Performs a prediction step of the model.
            validation_step(batch, batch_idx)
                Performs a validation step of the model.
            test_step(batch, batch_idx)
                Performs a test step of the model.
            configure_optimizers()
                Configures the optimizer for the model.
            on_save_checkpoint(checkpoint)
                Trims the model before saving it.
            load_state_dict(state_dict, strict=True)
                Loads the state dictionary of the model.  
                
        """




<a name="config"></a>
## Configuration  ⚙️
Okay.. all good till now, but of course you want to take some control. Don't you worry! A lot of the things you would want to do can simply be achieved by changing a single variable in the config! 

Two example configs are provided in 
* `/nlp/data/artemisp/multigpu-lm-templates/src/configs/train/llama_mrqa.py`
* `/nlp/data/artemisp/multigpu-lm-templates/src/configs/base.py`

## General Configuration
* `output_dir`: Specifies the current working directory of the project. This is used as a base to construct paths for data, outputs, and logs.
* `seed`: Sets a global seed for random number generators in PyTorch, NumPy, and Python's random module to ensure reproducible results. We use [`pl.seed_everything(cfg.seed, workers=True)`](https://pytorch-lightning.readthedocs.io/en/1.6.3/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything) to make sure that the experiments can be reproduced. It sets the random seeds for `numpy`, `pytorch`, and `random` globally.
* `debug`: A boolean flag that, when set to True, enables the output of detailed debug information during training.
* `strategy`: Defines the strategy for distributed training, e.g., 'ddp' for Distributed Data Parallel. This is crucial for multi-GPU or multi-node training.
* `metrics`: list of evaluate metrics for predictions
* `resume_from_checkpoint`: checkpoint to resume training or predict. 

## Training Configuration


### Prefix Tuning

Prefix Tuning is a parameter-efficient adaptation strategy for fine-tuning large language models. By prepending a sequence of trainable parameters (prefix) to the input, the model can adapt to new tasks with minimal updates to its pre-trained weights.

- **`prefix_tuning`**: A boolean flag indicating whether to enable Prefix Tuning. Setting this to `False` disables Prefix Tuning, while `True` activates it.
- **`prefix_tokens`**: Specifies the number of prefix tokens to use when Prefix Tuning is enabled. This value determines the length of the trainable prefix sequence.

### LoRA Configuration

Low-Rank Adaptation (LoRA) offers another approach to adapt pre-trained models to downstream tasks efficiently. LoRA introduces trainable low-rank matrices into specific layers of the model, allowing for task-specific adjustments without altering the original pre-trained parameters significantly.

- **`lora_config`**:
  - **`r`**: The rank of the low-rank matrices. A lower rank means fewer parameters to train, enhancing efficiency.
  - **`lora_alpha`**: Scaling factor for the LoRA matrices, controlling the magnitude of the adaptation.
  - **`lora_dropout`**: Dropout rate applied to the LoRA matrices, helping prevent overfitting.
  - **`bias`**: Specifies whether to adapt the bias terms. The value `"none"` indicates no adaptation of bias terms.
  - **`target_modules`**: A list of model components to which LoRA is applied. Common targets include query (`q_proj`), key (`k_proj`), value (`v_proj`) projections, and language model head (`lm_head`).
  - **`task_type`**: The type of task the model is being adapted to. `"CAUSAL_LM"` indicates a causal language modeling task, such as text generation.

### Quantization Configuration

Model Quantization reduces the precision of the model's parameters and computations, aiming to decrease model size and improve inference speed, especially on hardware with limited resources or specialized accelerators.

- **`quantization_config`**:
  - **`load_in_8bit`**: Enables loading model parameters in 8-bit precision, reducing memory usage during inference.
  - **`bnb_8bit_use_double_quant`**: Activates double quantization, a technique to further compress model weights.
  - **`bnb_8bit_quant_type`**: Specifies the quantization scheme. `"nf8"` refers to a novel 8-bit quantization format optimized for neural network weights.
  - **`bnb_8bit_compute_dtype`**: Determines the data type for computations. `"bfloat16"` offers a balance between precision and computational efficiency, suitable for many modern accelerators.

  Note that the quantization precision is defined in `datamodule_kwargs.quantization_precision` as an integer 4,8 or 16. The names above should match this precision or it will throw an error.


## Preprocessing Arguments

- **`preprocessing_kwargs`**:
    - **`remove_html`**: A boolean flag indicating whether HTML tags should be removed from the text data. Setting this to `False` retains HTML tags, which might be necessary for certain NLP tasks where HTML structure is relevant.
    
    - **`pad_punctuation`**: Determines whether spaces should be added around punctuation marks. This can help in tokenization processes but is disabled here (`False`) to maintain the original punctuation spacing.

    - **`drop_tables`**: A flag to specify if entries containing HTML tables should be excluded from the dataset. This is set to `False`, implying that data containing tables will not be discarded, which can be useful for tasks that require understanding or interpreting tabular data.

    - **`add_constants`**: Introduces constant key-value pairs to the dataset.  This can be particularly beneficial for instructional tasks or when a consistent prompt is needed to standardize responses.

    - **`column_dict`**: Maps the columns in the dataset to their roles as `inputs` or `targets` which are the keys of the dictionary mapped to lists of columns corresponding to inputs and targets for the model.

    - **`input_template`** and **`target_template`**: Define templates for formatting inputs and targets in the `column_dict``

    - **`concat_input_output`**: When set to `True`, this option indicates that inputs and outputs should be concatenated during preprocessing. This is often used in sequence-to-sequence models where the input and output form a continuous sequence.

    - **`keep_columns`**: Specifies which columns should be retained in the processed dataset. 


## Tokenization Arguments

- **`tokenization_kwargs`**:
    - **`tokenizer_name`**: Specifies the tokenizer to be used from huggingface, or local path.

    - **`max_input_length`** and **`max_target_length`**: Define the maximum number of tokens allowed for inputs and targets.

    - **`padding`**: Determines the padding strategy from [transformers.padding](https://huggingface.co/docs/transformers/en/pad_truncation)

    - **`truncation`**: Enabled (`True`) to cut down sequences exceeding the maximum length. 

    - **`concat_input_output`**: When set to `True`, indicates that the input and output sequences should be concatenated. This is particularly useful for models trained in a text-to-text format, where the task can benefit from viewing the input and output as a continuous sequence.

    - **`prefix_tuning`**: A flag derived from the broader configuration, indicating whether prefix tuning is to be applied during tokenization. Prefix tuning adds a sequence of trainable parameters at the beginning of inputs, allowing the model to adapt to specific tasks with minimal updates to its pre-trained weights.

    - **`n_prefix_tokens`**: Specifies the number of prefix tokens to use in conjunction with prefix tuning. This parameter is only relevant if `prefix_tuning` is enabled and allows for fine-tuning the extent of the prefix's influence on the model's performance.

    - **`decoder_prefix`**: A boolean indicating whether a prefix should also be added to the decoder inputs in sequence-to-sequence models. This is set to `False`, meaning that no additional tokens are prepended to the decoder inputs.

    - **`pad_token`**: Defines the token used for padding sequences up to the required length. Choosing an appropriate padding token is crucial for the model to correctly interpret padded areas of the input.



## DataModule Arguments

- **`datamodule_kwargs` 

    - **`debug`**: A boolean flag indicating whether to run the module in debug mode. This can increase the verbosity of logging to aid in troubleshooting and ensure the data pipeline is functioning as expected.

    - **`strategy`**: Specifies the distributed training strategy, such as 'ddp' for Distributed Data Parallel. This setting should align with the computational resources available and the training setup.

    - **`raw_data`**: A dictionary mapping dataset splits ('train', 'dev', etc.) to their respective file paths. Defined before.

    - **`deduplicate_columns`**: Lists the columns based on which deduplication should occur to avoid training on repeated data points. Typically used to ensure the uniqueness of entries.

    - **`load_from_cache_file`**: Determines whether to load preprocessed data from a cache, speeding up the initialization process when the data has been processed before.

    - **`num_workers`**: The number of subprocesses to use for data loading and preprocessing. A higher number can significantly reduce data preparation time.

    - **`batch_size`**: Specifies the number of data points processed in each batch, influencing memory usage and training dynamics.

    - **`shots`**: For few-shot learning scenarios, this limits the number of examples used from the training data, allowing for experimentation with smaller subsets.

    - **`dev_from_train`**: Controls how the validation (development) dataset is created. A value of `-1` uses a separate dev file, while other values indicate the number of samples to subsample from the training data.

    - **`overfit`**: When set to `True`, the model is trained and validated on the same dataset, useful for debugging and model architecture testing.

    - **`dev_size`**: Defines the size of the development set, either as an absolute number of samples or a fraction of the training set, depending on the context.

    - **`tiny`** and **`tiny_size`**: Enable training on a smaller subset of the data ("tiny dataset") for rapid prototyping or debugging, with `tiny_size` specifying the subset size.

    - **`filter_long_sequences`**: A boolean flag to enable the filtering of data points with sequences longer than specified in `tokenization_kwargs`, ensuring compatibility with model constraints.

    - **`preprocessing_kwargs`** and **`tokenization_kwargs`**: Dictionaries containing detailed configurations for preprocessing and tokenization steps, tailoring data preparation to the task and model requirements.

    - **`batch_tokenize`**: Indicates whether tokenization should be performed in batches for efficiency. This is particularly beneficial when dealing with large datasets.

    - **`predict_split`**: Specifies which data split should be used for prediction, allowing for flexibility in model deployment and testing.


## Generation Arguments
 - ** `generation_kwargs`: arguments passed in [`model.generate`](https://huggingface.co/docs/transformers/v4.38.2/en/model_doc/phi#transformers.PhiForCausalLM.generate)
    - **`logger_type`**: Specifies the logging backend to be used. Setting this to `'wandb'` indicates that Weights & Biases is chosen for logging the training runs, metrics, and potentially the model itself. Currently this is the only logging strategy we have implemented. Please make sure to run [`wandb init`](https://docs.wandb.ai/ref/cli/wandb-init) before submitting slurm jobs. 
    - **`name`**: A unique identifier for the current run or experiment. This name is used within WandB to differentiate between different training sessions. Here, `'llama2/lora/mrqa_10k'` suggests a naming convention that includes the model, technique, and dataset used.

    - **`save_dir`**: The local directory where WandB will save offline logs before syncing them to the cloud. This path is dynamically set based on the `OUTPUT_DIR` environment variable, with a fallback to a default directory (`wandb_logs`) within the project directory. This ensures that all logs are stored in a consistent location, regardless of the environment.

    - **`project`**: The name of the WandB project under which the run will be logged. This organizes runs into groups, making it easier to compare and analyze experiments. The project name can be set via the `WANDB_PROJ_NAME` environment variable, with a default fallback to `"test"`. This flexibility allows for seamless integration into different project setups and workflows.

    - **`log_model`**: Determines whether the model checkpoints are logged to WandB. Setting this to `False` disables model logging, which can be preferable for reducing bandwidth usage or when models are managed through another system. 

    - **`resume`**: Controls the behavior when resuming a previously interrupted WandB run. The default value `"allow"` enables automatic resumption based on the unique run ID saved in the WandB directory. This feature is crucial for long-running experiments that might be subject to interruptions, ensuring that logging continues seamlessly without creating duplicate entries.


 ## Optimizer Arguments
 - ** `optimizer_config`

## Module Arguments

The `module_kwargs` dictionary consolidates various settings and parameters critical for initializing and configuring the model within a PyTorch Lightning framework. This comprehensive guide breaks down each component to ensure a clear understanding of its purpose and impact on the model training and inference process.

### Core Model Settings

- **`model_name`**: Specifies the pre-trained model to be used.

- **`auto_model_class`**: Determines the class of model to be instantiated from the Transformers library. `"AutoModelForCausalLM"` is chosen for tasks that involve generating text based on prior context, such as story generation or conversational agents.

- **`prefix_tuning`** and **`n_prefix_tokens`**: Control the use of prefix tuning, a technique to adapt pre-trained models to new tasks by adding a fixed-length prefix to inputs. The number of prefix tokens is specified by `n_prefix_tokens`.

- **`initialize_from_vocab`**: A flag indicating whether the prefix tokens should be initialized from the model's vocabulary. Setting this to `False` means the initialization will likely be random or follow another specified strategy.

## Optimization and Precision

- **`optimizer`** and **`optimizer_type`**: Both set to `'AdamW'`, confirming the choice of optimizer for adjusting model weights during training. AdamW is known for its effectiveness in deep learning models.

- **`optimizer_config`**: Contains detailed settings for the optimizer, such as learning rate and weight decay, influencing the speed and stability of the training process.

- **`gradient_checkpointing`**: When enabled (`True`), reduces memory usage at the cost of slightly increased compute, by trading off between storing intermediate activations and recomputing them during backward passes.

- **`quantization_precision`** and **`precision`**: Define the numerical precision for model weights and computations. `quantization_precision` is set to `8` for 8-bit quantization, while `precision` is set to `"bf16"` (BFloat16), balancing performance and accuracy.

## Advanced Model Features

- **`lora`** and **`lora_config`**: Enable Low-Rank Adaptation (LoRA) and specify its configuration, allowing the model to adapt to new tasks with minimal additional parameters.

- **`quantization`** and **`quantization_config`**: Enable and configure model quantization to reduce the model size and potentially speed up inference, with settings like `load_in_8bit` to control how weights are stored and computed.

- **`tokenization_kwargs`**: Directs how inputs should be tokenized, affecting how data is prepared and fed into the model. Defined earlier. 

- **`generation_kwargs`**: Specifies parameters for the text generation process, such as the maximum number of new tokens to generate, offering control over the model's output. Defined earlier.

## Model Management

- **`freeze_encoder`**, **`freeze_encoder_layers`**, **`freeze_decoder`**, **`freeze_decoder_layers`**: Control which parts of the model are trainable. Freezing layers can speed up training and reduce overfitting by keeping pre-trained weights fixed.

- **`keep_in_fp32_modules`**: Lists model components that should remain in full precision (Float32), useful when training with mixed precision to maintain the accuracy of certain calculations.

- **`resume_from_checkpoint`**: Path to a checkpoint file from which training can be resumed, allowing for interrupted training processes to continue without loss of progress.

- **`postproc_fn`**: Specifies a function to be applied to the model's outputs for postprocessing, which can be necessary for tasks requiring specific output formatting or additional processing steps.

## Special for prediction
* `metrics`: list of [evaluate](https://huggingface.co/docs/evaluate/en/index) metrics for predictions
* `resume_from_checkpoint`: checkpoint to resume training or predict. 
* `raw_data = { "predict": f'file_to_predict'}`: defines prediction file
*  `datamodule_kwargs = { "predict_split": 'dev'}`: defines the prediction split from the input dataset


## Trainer Arguments
See [Pytorch Lightning Trainer API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api)



<a name="add_ds"></a>
# Special Topic: Adding your own dataset

* Step 1. Generate train.csv, and dev.csv files with your data. 
* Step 2. Update the config of your training/eval session as follows:
```
raw_data = {
    "train":  f'path/to/train.csv',
    "dev": f'path/to/dev.csv'
}
```
* Step 3. Define your preprocessing templates:
```
preprocessing_kwargs = {
    "remove_html": False,
    "pad_punctuation": False,
    "drop_tables": False,
    "add_constants": {"instruction": "In this task you are given a context and a question. You need to answer the question based on the context as briefly as possible."},
    "column_dict": {"inputs": ["instruction", "context", "question"], "target": "answer"},
    "input_template": "[INST] Context: {} Question: {} [/INST]",
    "target_template": "{}",
    "concat_input_output": True,
    "keep_columns": ["input", "target"],
}
```
The most important is the `column_dict`. It populates the `input_template`/`target_template` with the column values in order provided. 


<a name="train"></a>
# Ready to train?

You can submit a batch job:
`sbatch src/slurm_scripts/run_ft.sh --cfg path/to/your/config` to run on 8gpus. and
`sbatch src/slurm_scripts/run_ft_1gpu.sh --cfg path/to/your/config` to run on one. You can modify the scripts accordingly. 

For interractive debugging do the following:
` srun --gpus 1 --nodes 1 --mem-per-cpu 12GB  --constraint 48GBgpu --ntasks-per-node 1 --cpus-per-gpu 10 python  src/pl_ft.py --cfg /path/to/your/config`


<a name="evaluate"></a>
# Ready to evaluate?

Create a config which defines `resume_From_checkpoint` which is passed in the `module_kwargs` and also on the top level of the config. Then specify your `metrics` and `output_dir`. Finally, either pass a raw dataset with a `predict` split, or specify your prediction split in `datamodule_kwargs`. 

Run `sbatch src/slurm_scripts/run_predict_1gpu.sh --cfg path/to/your/config` to run on one gpu or select the 4gpu script for faster inference. You can modify the scripts accordingly. 


## How to Cite

If you use this project or software in your research or work, please consider citing it. 

```
@misc{apanagopoulou2024_parallelm,
  author = {Artemis Panagopoulou},
  title = {ParalleLM: Distributed Training Templates for NLP Tasks},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/artemisp/paralleLM}},
  commit = {abc123def},
  version = {1.0},
  doi = {},
}
```

## Requesting Features or Reporting Issues

We welcome contributions from the community! If you have suggestions for new features or have encountered any issues, please report them using our Issues page:

1. Navigate to the **Issues** tab in our project repository.
2. Click on **New Issue** to create a new issue.
3. Provide a detailed description of the feature request or the bug you encountered. Include any relevant details, screenshots, or steps to reproduce the issue.
4. Submit the issue.


## License

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

