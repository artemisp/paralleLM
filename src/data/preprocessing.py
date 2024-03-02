import os

import torch
from transformers import AutoTokenizer
from src.data.data_utils import (
    _remove_html, 
    _pad_punctuation, 
    _filter_na, 
    _filter_tables,
    _concat_text_input_output,
    _add_prefix_input_left_padding
    )

from dotenv import load_dotenv
load_dotenv(os.getcwd()+'/.env')
# Get the data root dir
HF_TOKEN = os.getenv('HF_ACCESS_TOKEN', None)

#####################################################################################
######################## Prepare Inputs/Labels DisentQA #############################
#####################################################################################
def get_inputs_and_targets(dataset, 
                           num_proc=1, 
                           load_from_cache_file=False, 
                           batch_tokenize=False,
                           **kwargs):
    """
    Processes a given dataset to structure it into a format suitable for a question-answering machine learning model. 

    Parameters:
    - dataset (Dataset): The dataset to process. Expected to be a Hugging Face `Dataset` object.
    - num_proc (int, optional): Number of processes to use in data mapping. Default is 1.
    - load_from_cache_file (bool, optional): Whether to load the processed dataset from a cache file. Default is False.
    - column_dict (dict, optional): A dictionary specifying the columns to be used as inputs and target. Default is {"inputs": ["text"], "target": "label"}
    - input_template (str, optional): Template string for formatting inputs. Default is "{}".
    - target_template (str, optional): Template string for formatting targets. Default is "{}".
    - remove_html (bool, optional): Flag to indicate whether HTML tags should be removed from the text. Default is True.
    - pad_punctuation (bool, optional): Flag to indicate whether to add spaces around punctuation. Default is False.
    - drop_tables (bool, optional): Flag to indicate whether to drop entries that contain HTML tables. Default is True.
    - drop_na (bool, optional): Flag to indicate whether to drop entries with missing data. Default is True.
    - add_constants (dict, optional): A dictionary specifying constant columns to add to the dataset. Default is None.
    - keep_columns (list, optional): A list of columns to keep in the dataset. Default is None.


    Returns:
    - Dataset: The processed dataset formatted according to the specified templates and flags.

    The function performs several steps:
    - Filters out entries with missing data.
    - Optionally filters out entries containing HTML tables.
    - Maps inputs and targets to the specified templates.
    - Optionally removes HTML tags and pads punctuation in the inputs and targets.

    """
    
    ## set up arguments
    column_dict=kwargs.get("column_dict", {"inputs": ["text"], "target": "label"})
    input_template=kwargs.get("input_template", "{}")
    target_template=kwargs.get("target_template", "{}")
    remove_html=kwargs.get("remove_html", True)
    pad_punctuation=kwargs.get("pad_punctuation", False)
    drop_tables=kwargs.get("drop_tables", True)
    drop_na=kwargs.get("drop_na", True)
                           
    
    # Function to map input data to a desired format.
    def map_inputs_labels(examples):
        # If the input is a string, convert it to a list for processing
        if isinstance(examples[column_dict["inputs"][0]], str):
            for k in column_dict["inputs"]:
                examples[k] = [examples[k]]
            if 'target' in column_dict:
                examples[column_dict["target"]] = [examples[column_dict["target"]]]
        
        # Format inputs and targets based on templates
        examples['input'] = [input_template.format(*[examples[k][i] for k in column_dict["inputs"]]) for i in range(len(examples[column_dict["inputs"][0]]))]
        if 'target' in column_dict:
            examples['target'] = [target_template.format(*[examples[k][i] for k in column_dict["target"]] if isinstance(column_dict["target"], list) else [examples[column_dict["target"]][i]]) for i in range(len(examples[column_dict["inputs"][0]]))]
        
        # Apply HTML removal and punctuation padding if enabled
        if remove_html:
            examples['input'] = [_remove_html(ex) for ex in examples['input']]
            if 'target' in column_dict:
                examples['target'] = [_remove_html(ex) for ex in examples['target']]
        if pad_punctuation:
            examples['input'] = [_pad_punctuation(ex) for ex in examples['input']]
            if 'target' in column_dict:
                examples['target'] = [_pad_punctuation(ex) for ex in examples['target']]
        return examples
    
    # Process dataset to remove examples with missing data
    if drop_na:
        print(f"Before drop na {len(dataset)}")
        dataset = dataset.map(_filter_na, num_proc=num_proc, load_from_cache_file=load_from_cache_file, desc="Marking NA")
        dataset = dataset.filter(lambda example: example['keep'], num_proc=num_proc, load_from_cache_file=load_from_cache_file, desc="Removing NA")
        dataset = dataset.remove_columns(['keep'])
        print(f"After drop na: {len(dataset)}")
    
    if kwargs.get("add_constants", False):
        print("Adding constant columns to dataset: " + str(kwargs['add_constants']))
        def add_constants(examples):
            for k,v in kwargs['add_constants'].items():
                examples[k] = v
            return examples
        dataset = dataset.map(add_constants, num_proc=num_proc, load_from_cache_file=load_from_cache_file, desc="Adding constants")
    
    
    # Mapping dataset to the format of inputs and targets
    print("Mapping dataset to inputs and targets....")
    dataset = dataset.map(map_inputs_labels, batched=True, num_proc=num_proc, load_from_cache_file=load_from_cache_file, desc="Mapping inputs and targets")
    
    # Process dataset to remove examples containing tables if enabled
    if drop_tables:
        print(f"Before drop tables {len(dataset)}")
        dataset = dataset.map(_filter_tables, num_proc=num_proc, load_from_cache_file=load_from_cache_file, desc="Marking tables")
        dataset = dataset.filter(lambda example: example['keep'], num_proc=num_proc, load_from_cache_file=load_from_cache_file, desc="Removing tables")
        dataset = dataset.remove_columns(['keep'])
        print(f"After drop tables: {len(dataset)}")
    
    # Drop unused columns
    columns_to_drop = [c for c in dataset.column_names if c not in kwargs.get('keep_columns', dataset.column_names)]
    print(f"Dropping columns {str(columns_to_drop)}")
    dataset = dataset.remove_columns(columns_to_drop)
    
    return dataset


def tokenize_inputs_and_targets(dataset, 
                                tokenizer, 
                                num_proc=1,
                                load_from_cache_file=False, 
                                **kwargs):
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if num_proc > 0 else "false"
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, token=HF_TOKEN, **kwargs)
        tokenizer.pad_token = getattr(tokenizer,kwargs.get("pad_token","pad_token"))
            
    
    predict=kwargs.get("predict", False)
    
    def tokenize(examples):        
        if kwargs.get("concat_input_output", False):
            tokenizer.padding_side='left'
            tokenizer.truncation_side='left'
            tokenized_input = tokenizer(examples['input'], 
                                      max_length=kwargs.get('max_input_length', 512), 
                                      padding=kwargs.get('padding', 'max_length'),
                                      truncation=kwargs.get('truncation', True),
                                      return_attention_mask=True, 
                                      add_special_tokens=False,
                                      return_tensors="pt"
                                      )
            tokenizer.padding_side='right'
            tokenizer.truncation_side='right'
            if 'target' in examples and not predict:
                tokenized_target = tokenizer([t+tokenizer.eos_token for t in examples['target']], 
                                max_length=kwargs.get('max_target_length', 30), 
                                padding=kwargs.get('padding', 'max_length'),
                                truncation=kwargs.get('truncation', True),
                                return_attention_mask=True, 
                                add_special_tokens=False,
                                return_tensors="pt"
                                )
                if kwargs.get('prefix_only', False):
                    llm_tokens, input_part_targets_len = _concat_text_input_output(torch.tensor([[] for _ in range(tokenized_input.input_ids.shape[0])]), 
                                                            torch.tensor([[] for _ in range(tokenized_input.input_ids.shape[0])]), 
                                                            tokenized_target.input_ids, 
                                                            tokenized_target.attention_mask
                                                            )
                else:
                    llm_tokens, input_part_targets_len = _concat_text_input_output(tokenized_input.input_ids,
                                                                tokenized_input.attention_mask, 
                                                                tokenized_target.input_ids, 
                                                                tokenized_target.attention_mask
                                                                )
                labels = llm_tokens['input_ids'].masked_fill(
                    llm_tokens['input_ids'] == tokenizer.pad_token_id, -100
                    )
            
                # do not apply loss to the text input part
                for i, l in enumerate(input_part_targets_len):
                    labels[i][:l] = -100
                examples['labels'] = labels 
                examples['input_ids'] = llm_tokens['input_ids']  
                examples['attention_mask'] = llm_tokens['attention_mask'] 
            else:
                examples['input_ids'] = tokenized_input["input_ids"]
                examples['attention_mask'] = tokenized_input["attention_mask"]
            
        else:
            tokenizer.padding_side='right'
            tokenizer.truncation_side='right'
            tokenized_input = tokenizer(examples['input'], 
                                      max_length=kwargs.get('max_input_length', 512), 
                                      padding=kwargs.get('padding', 'max_length'),
                                      truncation=kwargs.get('truncation', True),
                                      return_attention_mask=True, 
                                      add_special_tokens=True,
                                      return_tensors="pt"
                                      )
            if 'target' in examples and not predict:
                tokenized_target = tokenizer([t for t in examples['target']], 
                                max_length=kwargs.get('max_target_length', 30), 
                                padding=kwargs.get('padding', 'max_length'),
                                truncation=kwargs.get('truncation', True),
                                return_attention_mask=True, 
                                add_special_tokens=False,
                                return_tensors="pt"
                                )
                examples['labels'] = tokenized_target["input_ids"]
                examples['labels'] = examples['labels'].masked_fill(
                     examples['labels'] == tokenizer.pad_token_id, -100
                    )
                    
            examples['input_ids'] = tokenized_input["input_ids"]
            examples['attention_mask'] = tokenized_input["attention_mask"]

        
        if kwargs.get('prefix_input_left_padding', False):
            n_prefix_tokens = kwargs.get('n_prefix_tokens', 30)
            if predict:
                 examples['input_ids'], examples['attention_mask'] = _add_prefix_input_left_padding(examples['input_ids'], 
                                                                                       )                 
            else:
                examples['input_ids'] = torch.cat([torch.full((1,n_prefix_tokens),kwargs.get('prefix_token_id', 50256)).repeat(examples['input_ids'].shape[0], 1).type_as(examples['input_ids']).long(), examples['input_ids']], 1)
                examples['attention_mask'] = torch.cat([torch.full((1,n_prefix_tokens), 1).repeat(examples['attention_mask'].shape[0], 1).type_as(examples['input_ids']).long(), examples['attention_mask']], 1)
                if 'target' in examples and not predict:
                    examples['labels'] = torch.cat([torch.full((1, n_prefix_tokens), -100).repeat(examples['input_ids'].shape[0], 1).type_as(examples['input_ids']).long(),examples['labels']], 1)
            # for encoder-decoder models, like T5            
            if kwargs.get('decoder_prefix', False):
                examples['decoder_input_ids'] = torch.full((examples['input_ids'].size(0), n_prefix_tokens), kwargs.get('prefix_token_id', 50256)).type_as(examples['input_ids']).long()
        
        return examples
    dataset = dataset.map(tokenize, batched=kwargs.get('batched', True), num_proc=num_proc, load_from_cache_file=load_from_cache_file)
    return dataset


def batch_tokenize_inputs_and_targets(batch, 
                                tokenizer, 
                                num_proc=1,
                                load_from_cache_file=False, 
                                encoder=None,
                                context_aware_preprocessing_config={},
                                **kwargs):
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, token=HF_TOKEN, **kwargs)
        tokenizer.pad_token = getattr(tokenizer,kwargs.get("pad_token","pad_token"))
            
    
    predict=kwargs.get("predict", False)
    
    def tokenize(examples):        
        if kwargs.get("concat_input_output", False):
            tokenizer.padding_side='left'
            tokenizer.truncation_side='left'
            tokenized_input = tokenizer(tokenizer.bos_token + examples['input'], 
                                      max_length=kwargs.get('max_input_length', 512), 
                                      padding=kwargs.get('padding', 'max_length'),
                                      truncation=kwargs.get('truncation', True),
                                      return_attention_mask=True, 
                                      add_special_tokens=False,
                                      return_tensors="pt"
                                      )
            tokenizer.padding_side='right'
            tokenizer.truncation_side='right'
            if 'target' in examples and not predict:
                tokenized_target = tokenizer(examples['target'] + tokenizer.eos_token, 
                                max_length=kwargs.get('max_target_length', 30), 
                                padding=kwargs.get('padding', 'max_length'),
                                truncation=kwargs.get('truncation', True),
                                return_attention_mask=True, 
                                add_special_tokens=False,
                                return_tensors="pt"
                                )
                if kwargs.get('prefix_only', False):
                    llm_tokens, input_part_targets_len = _concat_text_input_output(torch.tensor([[] for _ in range(tokenized_input.input_ids.shape[0])]), 
                                                            torch.tensor([[] for _ in range(tokenized_input.input_ids.shape[0])]), 
                                                            tokenized_target.input_ids, 
                                                            tokenized_target.attention_mask
                                                            )
                else:
                    llm_tokens, input_part_targets_len = _concat_text_input_output(tokenized_input.input_ids,
                                                                tokenized_input.attention_mask, 
                                                                tokenized_target.input_ids, 
                                                                tokenized_target.attention_mask
                                                            )
     
                labels = llm_tokens['input_ids'].masked_fill(
                    llm_tokens['input_ids'] == tokenizer.pad_token_id, -100
                    )
            
                # do not apply loss to the text input part
                for i, l in enumerate(input_part_targets_len):
                    labels[i][:l] = -100
                                                                        
                examples['labels'] = labels 
                examples['input_ids'] = llm_tokens['input_ids']  
                examples['attention_mask'] = llm_tokens['attention_mask'] 
            else:
                examples['input_ids'] = tokenized_input["input_ids"]
                examples['attention_mask'] = tokenized_input["attention_mask"]
            
        else:
            tokenizer.padding_side='right'
            tokenizer.truncation_side='right'
            tokenized_input = tokenizer(tokenizer.bos_token + examples['input'], 
                                      max_length=kwargs.get('max_input_length', 512), 
                                      padding=kwargs.get('padding', 'max_length'),
                                      truncation=kwargs.get('truncation', True),
                                      return_attention_mask=True, 
                                      add_special_tokens=True,
                                      return_tensors="pt"
                                      )
            if 'target' in examples and not predict:
                tokenized_target = tokenizer(examples['target'] + tokenizer.eos_token, 
                                max_length=kwargs.get('max_target_length', 30), 
                                padding=kwargs.get('padding', 'max_length'),
                                truncation=kwargs.get('truncation', True),
                                return_attention_mask=True, 
                                add_special_tokens=False,
                                return_tensors="pt"
                                )
                examples['labels'] = tokenized_target["input_ids"]
                examples['labels'] = examples['labels'].masked_fill(
                     examples['labels'] == tokenizer.pad_token_id, -100
                    )
            
            examples['input_ids'] = tokenized_input["input_ids"]
            examples['attention_mask'] = tokenized_input["attention_mask"]

        
        if kwargs.get('prefix_tuning', False):
            n_prefix_tokens = kwargs.get('n_prefix_tokens', 30)
            if kwargs.get('prefix_input_left_padding', False):
                 examples['input_ids'], examples['attention_mask'] = _add_prefix_input_left_padding(examples['input_ids'], 
                                                                                                    examples['attention_mask'],                                                                                    )
                 
            else:
                examples['input_ids'] = torch.cat([torch.full((1,n_prefix_tokens),kwargs.get('prefix_token_id', 50256)).repeat(examples['input_ids'].shape[0], 1).type_as(examples['input_ids']).long(), examples['input_ids']], 1)
                examples['attention_mask'] = torch.cat([torch.full((1,n_prefix_tokens), 1).repeat(examples['attention_mask'].shape[0], 1).type_as(examples['input_ids']).long(), examples['attention_mask']], 1)
                if 'target' in examples and not predict:
                    examples['labels'] = torch.cat([torch.full((1, n_prefix_tokens), -100).repeat(examples['input_ids'].shape[0], 1).type_as(examples['input_ids']).long(),examples['labels']], 1)
            # for encoder-decoder models, like T5            
            if kwargs.get('decoder_prefix', False):
                examples['decoder_input_ids'] = torch.full((examples['input_ids'].size(0), n_prefix_tokens), kwargs.get('prefix_token_id', 50256)).type_as(examples['input_ids']).long()
        
        return examples


            
    return tokenize(batch)