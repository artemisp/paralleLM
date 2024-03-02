import re
import string
import torch

# Function to add spaces around punctuation.
def _pad_punctuation(text):
    pattern = r'([{punct}])'.format(punct=string.punctuation)
    text = re.sub(pattern, r' \1 ', text)  # Add space around each punctuation mark
    text.replace("?","")  # Remove question marks
    text = re.sub(r'\s+', ' ', text)  # Collapse consecutive spaces
    return text

# Function to remove HTML tags from text.
def _remove_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)  # Remove anything that looks like an HTML tag


# Function to filter out examples with missing data
def _filter_na(examples):
    if not isinstance(examples, list):
        examples = [examples]
    for example in examples:
        example['keep'] = True
        for key, value in example.items():
            if value is None or (isinstance(value, str) and len(value) == 0):
                example['keep'] = False
    return examples if len(examples) > 1 else examples[0]

# Function to filter out examples containing HTML tables
def _filter_tables(examples):
    if not isinstance(examples, list):
        examples = [examples]
    for example in examples:
        example['keep'] = True
        if '<table>' in example['input'].lower():
            example['keep'] = False
    return examples if len(examples) > 1 else examples[0]


#adapted from: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_vicuna_instruct.py
def _concat_text_input_output(input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][input_ids[i].shape[0]-this_input_ones:],
                    output_ids[i],
                    input_ids[i][:input_ids[i].shape[0]-this_input_ones]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][input_ids[i].shape[0]-this_input_ones:],
                    output_atts[i],
                    input_atts[i][:input_ids[i].shape[0]-this_input_ones]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len


def _add_prefix_input_left_padding(input_ids, input_atts,  n_prefix_tokens=30, pad_token_id=0, prefix_token_id=50256):
    llm_tokens = {"input_ids": [], "attention_mask": []}
    for i in range(input_ids.size(0)):
        this_input_ones = input_atts[i].sum()
        this_pad_len = input_ids[i].shape[0]-this_input_ones
        llm_tokens['input_ids'].append(
            torch.cat([
                input_ids[i][:this_pad_len],
                torch.full((n_prefix_tokens,),prefix_token_id), # prefix
                input_ids[i][this_pad_len:], # input
            ])
        )
        llm_tokens['attention_mask'].append(
            torch.cat([
                input_atts[i][:this_pad_len], # padding
                torch.full((n_prefix_tokens,),1), # prefix
                input_atts[i][this_pad_len:], # input
            ])
        )
                        
    llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
    llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
    return llm_tokens['input_ids'], llm_tokens['attention_mask']
