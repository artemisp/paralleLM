from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import datasets
import copy
import random
import os
import sys
sys.path.append(os.getcwd())
from src.data.preprocessing import *
from dotenv import load_dotenv
load_dotenv(os.getcwd()+'/.env')
# Get the data root dir
cache_dir = os.getenv('CACHE_DIR')

class QADataset(Dataset):
    def __init__(self, dataset, tokenizer,cfg):
        self.tokenizer = tokenizer
        self.data = get_inputs_and_targets(dataset,
                                        use_context=cfg.use_context, 
                                        load_from_cache_file=not (cfg.debug or cfg.tiny or cfg.shots > 0))  # datasets.Dataset object
        self.cfg =cfg
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        input_tensors = self.tokenizer(sample['input'], max_length=self.cfg.input_max_length,
                                         padding="max_length",
                                         truncation=True, return_attention_mask=True, add_special_tokens=True,
                                         return_tensors="pt")
        target_tensors = self.tokenizer(sample['target'], max_length=self.cfg.output_max_length, padding="max_length",
                                        truncation=True, return_attention_mask=True, add_special_tokens=True,
                                        return_tensors="pt")

        input_ids = input_tensors["input_ids"].flatten()
        attention_mask = input_tensors["attention_mask"].flatten()
        labels = target_tensors["input_ids"].flatten()
        labels[labels == 0] = -100
        
        sample['input_ids'] = input_ids
        sample['attention_mask'] = attention_mask
        sample['labels'] = labels

        return sample

class QADataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule for processing question-answering data.

    Args:
        dataset (Dataset): A preprocessed dataset in the form of `datasets.Dataset` to process.
        tokenizer: The tokenizer object to use for tokenization.
        cfg: The configuration object that contains the required parameters for processing the dataset.
        batch_size (int, optional): The batch size to use for training and inference. Defaults to `None`.
    """
    def __init__(self, dataset: datasets.Dataset, tokenizer, cfg, batch_size=None):
        super().__init__()
        ## required for autotuning of batch size
        if batch_size: 
            self.batch_size = batch_size
        else:
            self.batch_size = cfg.batch_size
        
        if cfg.tiny:
            self.dataset = dataset
            self.dataset['train'] = self.dataset["train"].shuffle(seed=cfg.seed).select(range(cfg.tiny_size))
            self.dataset['validation'] = self.dataset["validation"].shuffle(seed=cfg.seed).select(range(cfg.tiny_size))
        else:
            if cfg.dev_from_train > 0:
                # don't shuffle as not to get context overlap
                # TODO: make sure there is no overlap in contexts. 
                self.dataset = dataset['train'].train_test_split(test_size=cfg.dev_from_train,shuffle=False, seed=cfg.seed)
                self.dataset['validation'] = self.dataset['test']
                # assert len(set(self.dataset['train'].unique('context')).intersection\
                #     (set(self.dataset['validation'].unique('context')))) == 0, "There is train-dev overlap."
                self.dataset['test'] = copy.deepcopy(dataset['validation'])
            else:
                self.dataset = dataset
                self.dataset['test'] = self.dataset['validation']
        if cfg.shots > 0:
            self.dataset['train'] = dataset["train"].shuffle(seed=cfg.seed).select(range(cfg.shots))
        if cfg.overfit:
            self.dataset['validation'] = copy.deepcopy(self.dataset['train'] ) 
        
        if 'test' not in self.dataset:
            self.dataset['test'] = copy.deepcopy(self.dataset['validation'])
            
        self.tokenizer = tokenizer
        self.cfg=cfg

    def setup(self, stage=None):
        """
        Set up the training, validation, and testing datasets.

        Args:
            stage (str, optional): The stage to set up. Defaults to `None`. It is used for distributed processing.
        """
        print("Setting up dataloaders...")
        self.train_dataset = QADataset(self.dataset['train'], self.tokenizer, self.cfg)
        self.dev_dataset = QADataset(self.dataset['validation'], self.tokenizer, self.cfg)
        self.test_dataset = QADataset(self.dataset['test'], self.tokenizer, self.cfg)
        
        if self.cfg.debug:
            print("Random train sample")
            randidx = random.randrange(0,len(self.train_dataset))
            print("\n".join([f"{k}:{v}" for k,v in self.train_dataset[randidx].items()]))
            print("Random dev sample")
            randidx = random.randrange(0,len(self.dev_dataset))
            print("\n".join([f"{k}:{v}" for k,v in self.dev_dataset[randidx].items()]))
        print(f"Loaded {len(self.train_dataset)} train, {len(self.dev_dataset)} dev, {len(self.test_dataset)} test")
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.cfg.num_workers)

