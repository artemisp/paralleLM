import re

#####################################################################################
######################## Prepare Inputs/Labels Base #################################
#####################################################################################
#### google-research/text-to-text-transfer-transformer/t5/data/preprocessors.py #####
#####################################################################################
def get_inputs_and_targets(dataset, use_context=True, num_proc=1, load_from_cache_file=False, column_dict={"context":"context", "question":"question", "answer": "answer"}):
    """
    Augment a preprocessed (flattened) MRQA NQ dataset with a inputs and labels for T5 model.
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
    """  
    def _string_join(lst):
        # Join on space, but collapse consecutive spaces.
        out = ' '.join(lst)
        out = re.sub(r'\s+', ' ', out)
        return out

    def _pad_punctuation(text):
        """Adds spaces around punctuation."""
        # Add space around punctuation.
        text = re.sub(r'([[:punct:]])', r' \1 ', text)
        # Collapse consecutive whitespace into one space.
        text = re.sub(r'\s+', ' ', text)
        return text
    
    Q = column_dict["question"]
    C = column_dict["context"]
    A = column_dict["answer"]
    def map_inputs_labels(examples):
        if use_context:
            examples['input'] = [
                _string_join(['question:', _pad_punctuation(q), 'context:', _pad_punctuation(c)]) 
                    for q,c in zip(examples[Q], examples[C])
                    ]
        else:
            examples['input'] = [
                _string_join(['question:', _pad_punctuation(q)]) 
                    for q in examples[Q]
                    ]
        examples['target'] = [
            _string_join(['answer:', _pad_punctuation(a)]) 
                for a in examples[A]
                ]
        return examples
    print("Mapping dataset to inputs and targets for T5....")
    dataset = dataset.map(map_inputs_labels, batched=True, num_proc=num_proc, load_from_cache_file=False)
    return dataset
