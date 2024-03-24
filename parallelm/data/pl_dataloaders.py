from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AutoTokenizer
import transformers
import pytorch_lightning as pl
import datasets
import copy
import random
import os
import sys
import json
import torch

from dotenv import load_dotenv
load_dotenv(os.getcwd()+'/.env')
# Get the data root dir
cache_dir = os.getenv('CACHE_DIR', "./.cache")

from parallelm.data.preprocessing import get_inputs_and_targets, tokenize_inputs_and_targets, batch_tokenize_inputs_and_targets


class CustomDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, split, tokenizer = None, **kwargs):
        """
        Lightning data module for data processing.
        
        Args:
            dataset (Dataset): A preprocessed dataset in the form of `datasets.Dataset` to process.
            tokenizer: The tokenizer object to use for tokenization.
            cfg: The configuration object that contains the required parameters for processing the dataset.
        Keyword Args:
            batch_tokenize (bool): Whether to batch tokenize the data, if set to False the entire dataset is preloaded in memory
            num_workers (int): The number of workers to use for processing the data.
            load_from_cache_file (bool): Whether to load the data from the cache file.
            tokenization_kwargs (dict): The tokenization arguments to use for tokenizing the data. More specifically,
                - padding (str): The padding strategy to use for the sequences.
                - truncation (str): The truncation strategy to use for the sequences.
                - max_length (int): The maximum length of the input sequence.
                - max_target_length (int): The maximum length of the target sequence.  
        """
        self.data = dataset
        self.kwargs = kwargs
        self.tokenizer = tokenizer
        self.split = split

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Gets an item from the dataset by index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input_ids, attention_mask, and labels tensors for the specified index.
        """
        sample = self.data[index]
        if self.kwargs.get("batch_tokenize", False):
            sample = batch_tokenize_inputs_and_targets(sample, 
                                                self.tokenizer,
                                                num_proc=self.kwargs.get('num_workers', 1), 
                                                load_from_cache_file=False, 
                                                predict=self.split=='predict',
                                                # encoder=self.encoder if self.kwargs.get('preprocessing_kwargs', {}).get("context_aware_prefix", False) else None,
                                                **self.kwargs.get('tokenization_kwargs', {})
                                                )
        if self.split != 'train':
            return sample
        # return indices to track in training
        return index, sample

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer=None, batch_size=None, **kwargs):
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
        super().__init__()
        
        self.debug = kwargs.get('debug', False)
        self.strategy = kwargs.get('strategy', '')
        
        ## required for autotuning of batch size
        self.batch_size = batch_size if batch_size else kwargs.get('batch_size', 8)
        self.num_workers = kwargs.get('num_workers', 4)
        
        if tokenizer == None:
            tokenizer = kwargs.get('tokenization_kwargs', {}).get('tokenizer_name', 't5-small')
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer,token=os.environ.get('HF_ACCESS_TOKEN', None), **kwargs.get('tokenization_kwargs', {}), cache_dir=cache_dir) if isinstance(tokenizer, str) else tokenizer 
            self.tokenizer.pad_token = getattr(self.tokenizer,kwargs.get('tokenization_kwargs', {}).get("pad_token","pad_token"))
            self.tokenizer.pad_token_id = getattr(self.tokenizer,f'{kwargs.get("tokenization_kwargs", {}).get("pad_token","pad_token")}_id')
        else:
            self.tokenizer = tokenizer
        
        self.sampler_state = None 
        
        self.kwargs = kwargs
        self.current_epoch = 0
        self.global_step = 0
        self.effective_batch_size = self.batch_size * max(1, torch.cuda.device_count()) if 'dp' in self.strategy else self.batch_size

            
    def preprocess_datasets(self):
        kwargs = self.kwargs
        self.dataset = self.get_raw_dataset(**kwargs)
        
        self.splits = kwargs.get('splits', list(self.dataset.keys()))

        if kwargs.get("deduplicate_columns", []):

            def get_hash(example, col_name):
                """Get hash of content field."""
                return {"hash": example[col_name]} # can use any hashing function here
                
            def check_uniques(example, uniques):
                """Check if current hash is still in set of unique hashes and remove if true."""
                if example["hash"] in uniques:
                    uniques.remove(example["hash"])
                    return True
                else:
                    return False
            for col_name in kwargs.get("deduplicate_columns", []):
                for split in self.splits:
                    self.dataset[split] = self.dataset[split].map(get_hash, fn_kwargs={"col_name": col_name}, num_proc=kwargs.get('num_workers', 1), load_from_cache_file=kwargs.get('load_from_cache_file', False), desc=f"Hashing {col_name} for {split} dataset")
                    uniques = set(self.dataset[split].unique("hash"))
                    self.dataset[split] = self.dataset[split].filter(check_uniques, fn_kwargs={"uniques": uniques}, num_proc=kwargs.get('num_workers', 1), load_from_cache_file=kwargs.get('load_from_cache_file', False), desc=f"Deduplicating {col_name} for {split} dataset")
                    
        ## preprocessing
        for split in self.splits:
            self.dataset[split] = get_inputs_and_targets(self.dataset[split], 
                                                         num_proc=kwargs.get('num_workers', 1), 
                                                         load_from_cache_file=kwargs.get('load_from_cache_file', False), 
                                                         batch_tokenize=kwargs.get('batch_tokenize', False),
                                                         **kwargs.get('preprocessing_kwargs', {})
                                                         )

        ## filtering
        if kwargs.get('filter_long_sequences', False):
            self.filter_long_sequences(**kwargs)
        
        ## dev setup             
        self.configure_data_splits(**kwargs)
        
        ## tokenization
        if not kwargs.get('batch_tokenize', False):
            for split in self.splits:
                self.dataset[split] = tokenize_inputs_and_targets(self.dataset[split], 
                                                                self.tokenizer,
                                                                num_proc=kwargs.get('num_workers', 1), 
                                                                load_from_cache_file=kwargs.get('load_from_cache_file', False), 
                                                                predict=split=='predict',
                                                                **kwargs.get('tokenization_kwargs', {})
                                                                )
        
   
    def configure_data_splits(self, **kwargs):
        
        if 'predict' in self.splits:
            if kwargs.get("predict_size", -1)>0:
                self.dataset['predict'].shuffle().select(range(kwargs.get("predict_size", -1)))
                
        # if training file is not provided it is predict setup
        if 'train' not in self.splits:
            return
        
        ## dev data
        if isinstance(kwargs.get('dev_size', -1), float) and kwargs.get('dev_size', .1) <= 1.:
            dev_size = int(len(self.dataset['train'])*kwargs.get('dev_size'))
        else:
            dev_size = kwargs.get('dev_size', -1)
        if "dev" in self.splits:
            self.dataset['dev'] = self.dataset['dev'].select(range(dev_size))
        if kwargs.get('dev_from_train', -1) > 0:
            self.dataset = self.dataset['train'].train_test_split(test_size=dev_size,shuffle=False)
            self.dataset['dev'] = self.dataset['test']
    
        
        ## training size 
        if kwargs.get('tiny', False): 
            for split in self.splits:
                self.dataset[split] = self.dataset["train"].shuffle().select(range(kwargs.get('tiny_size', 1024)))
            
        elif kwargs.get('shots', -1) > 0:
            self.dataset['train'] = self.dataset["train"].shuffle().select(range(kwargs.get('shots', 1024)))
        
        ## overfitting
        if kwargs.get('overfit', True):
            self.dataset['dev'] = copy.deepcopy(self.dataset['train']) 
 
        
    def get_raw_dataset(self, **kwargs):
        ## raw data
        if isinstance(kwargs.get('raw_data', None), str):
            raw_dataset = datasets.load_dataset(kwargs['raw_data'], cache_dir=cache_dir)
            
            if kwargs.get('id2label', None):
                # replace the label ids with the label names
                def convert_label(examples):
                    examples['word_label'] = [kwargs['id2label'][l] for l in examples['label']]
                    return examples
                self.splits = kwargs.get('splits', list(raw_dataset.keys()))
                for split in self.splits:
                    raw_dataset[split] = raw_dataset[split].map(convert_label, batched=True, load_from_cache_file=kwargs.get('load_from_cache_file', False), num_proc=kwargs.get('num_workers', 1), desc=f"Converting label ids to names for {split} dataset")
            
        elif isinstance(kwargs.get('raw_data', None), datasets.Dataset):
            raw_dataset = kwargs['raw_data']
            
        elif isinstance(kwargs.get('raw_data', None), dict):
            # TODO: add support for other file types
            if 'csv' in list(kwargs['raw_data'].values())[0]:
                raw_dataset = datasets.load_dataset('csv', data_files=kwargs['raw_data'], delimiter=',', cache_dir=cache_dir)
            elif 'json' in list(kwargs['raw_data'].values())[0]:
                splits = kwargs.get('splits', list(kwargs['raw_data'].keys()))
                raw_dataset = {}
                for split in kwargs.get('splits', list(kwargs['raw_data'].keys())):
                    data_dict = json.load(open(kwargs['raw_data'][split]))
                    raw_dataset[split] = datasets.Dataset.from_dict(data_dict)                
        else:
            raise ValueError("raw_data must be a path to a dataset, a dataset object, or a hugginface dataset name")
        
        # we expect validation split to be named "dev"
        POSSIBLE_VAL_NAMES = ['validation', 'val', 'eval', "development", "evaluation", "validate", "evaluate"]
        for val_name in POSSIBLE_VAL_NAMES:
            if val_name in raw_dataset.keys():
                raw_dataset['dev'] = raw_dataset[val_name]
                del raw_dataset[val_name]
                
        if kwargs.get('predict_split', None):
            raw_dataset['predict'] =  raw_dataset[kwargs['predict_split']]
        return raw_dataset
    
    def filter_long_sequences(self, **kwargs):
        input_max_length = kwargs.get('tokenization_kwargs', {}).get('max_output_length', 512)
        target_max_length = kwargs.get('tokenization_kwargs', {}).get('max_target_length', 30)

        print(f"Filtering input sequences longer than {input_max_length} and output sequences longer than {target_max_length}....")
        def filter_long_sequences(example):
            return len(self.tokenizer(example['input'], add_special_tokens=True)['input_ids']) <= input_max_length \
                and ('target' not in example or ('target' in example and len(self.tokenizer(example['target'], add_special_tokens=True)['input_ids']) <= target_max_length))
        for split in self.splits:
            print(f"{split} size before filtering long sequences is {len(self.dataset[split])}")
            self.dataset[split] = self.dataset[split].filter(filter_long_sequences, num_proc=kwargs.get('num_workers', 1), load_from_cache_file=kwargs.get('load_from_cache_file', False), desc=f"Filtering long sequences for {split} dataset")
            print(f"{split} size after filtering long sequences is {len(self.dataset[split])}")
                
    def setup(self, stage=None):
        """
        Set up the training, dev, and testing datasets.

        Args:
            stage (str, optional): The stage to set up. Defaults to `None`. It is used for distributed processing.
        """
        print("Preprocessing data...")
        self.preprocess_datasets()
        # Make sure all processes wait until preprocessing is done
        if 'dp' in self.strategy and torch.cuda.device_count() > 1:
            torch.distributed.barrier()
            
        print("Setting up dataloaders...")
        print(self.splits)
        for split in self.splits:
            setattr(self, f"{split}_dataset", CustomDataset(self.dataset[split], split=split, tokenizer=self.tokenizer, **self.kwargs))
        
            if 'dp' in self.strategy and torch.cuda.device_count() > 1:
                setattr(self, f"{split}_sampler", DistributedSampler(getattr(self, f"{split}_dataset"), shuffle= split == 'train'))
            else:
                setattr(self, f"{split}_sampler", RandomSampler(getattr(self, f"{split}_dataset")) if split == 'train' else SequentialSampler(getattr(self, f"{split}_dataset")))
                
        if self.debug:
            for split in self.splits:
                print(f"Random {split} sample")
                randidx = random.randrange(0,len(self.dataset[split]))
                print("\n".join([f"{k}:{v}" for k,v in self.dataset[split][randidx].items()]))
                
        for split in self.splits:
            print(f"Loaded {len(getattr(self, f'{split}_dataset'))} datasaet samples for {split} split")
        
    def train_dataloader(self):
        if 'dp' in self.strategy and torch.cuda.device_count() > 1:
            self.train_sampler.set_epoch(self.current_epoch)
        return DataLoader(self.train_dataset, sampler=self.train_sampler,  batch_size=self.batch_size, num_workers=self.num_workers,  pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset,sampler=self.dev_sampler, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False,  pin_memory=False)
            
    def test_dataloader(self):
        return DataLoader(self.test_dataset,sampler=self.test_sampler, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False,  pin_memory=False)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,sampler=self.predict_sampler, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False, pin_memory=False)