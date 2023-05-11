from tqdm import tqdm
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from datasets import load_dataset
import pytorch_lightning as pl
from transformers import AutoTokenizer
import json
import os
import sys
import ast
sys.path.append(os.getcwd())

from src.data.preprocessing import get_inputs_and_targets
from src.predict.qa_utils import *
from src.data.pl_dataloaders import *
from src.data.postprocessing import *
from src.data.preprocessing import *
import src.pl_modules as pl_modules



from dotenv import load_dotenv
from mmengine.config import DictAction, Config


# Load the variables from the .env file
load_dotenv(os.getcwd()+'/.env')
# Get the data root dir
data_base_dir = os.getenv('FAITHFULNESS_DATA_DIR')
prediction_dir = os.getenv('PREDICTION_DIR')
cache_dir = os.getenv('CACHE_DIR')



#####################################
########## Arguments ################
###################################

parser = argparse.ArgumentParser()
parser.add_argument('--cfg',default=f'{os.getcwd()}/src/configs/t5/configs/base.py',  help='path to config file')
parser.add_argument('--use_checkpoint',default=f'',type=str,  help='Use specific chekckpoint. If empty will generate from pretrained cfg.model_name')
parser.add_argument('--predict_csv_files', nargs='+', default=[], help='CSV files to make predictions')
parser.add_argument('--input_to_column_dict', default='{"question": "question", "context": "context"}', type=str, help="Json that maps context and question to corresponding csv columns. If context is empty then not used")
parser.add_argument('--target_to_column_dict', default='{"answer": "answer"}', type=str, help="Json that maps answer to corresponding csv column.")
parser.add_argument('--pred_column', default='prediction', type=str, help="Name of column to place predictions.")
parser.add_argument('--cfg-options',
                    nargs='+',
                    action=DictAction,
                    metavar="KEY=VALUE",
                    help='overwrite parameters in cfg from commandline')
parser.add_argument('--num_workers', default=1,type=int,  help='number of cpuss')

args = parser.parse_args()


num_workers = args.num_workers

cfg = Config.fromfile(args.cfg)
if args.cfg_options:
    cfg.merge_from_dict(args.cfg_options)
print(cfg)
    
predict_csv_files = args.predict_csv_files
column2name = json.loads(args.input_to_column_dict)
column2name.update(json.loads(args.target_to_column_dict))
use_checkpoint = args.use_checkpoint
pred_column = args.pred_column

#####################################
########## SETUP ################
###################################

os.environ["TOKENIZERS_PARALLELISM"] = "true" if num_workers > 0 else "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["TRANSFORMERS_CACHE"]  = os.getenv('CACHE_DIR')


if '16' in cfg.precision:
    torch.set_float32_matmul_precision('medium')
pl.seed_everything(cfg.seed, workers=True)


################################
########## DATA ################
################################

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
def tokenize(examples):
    tokenized = tokenizer(examples['input'],
              max_length=cfg.input_max_length,
              padding="max_length",
              truncation=True, 
              return_attention_mask=True, 
              add_special_tokens=True,
            return_tensors="pt")
    examples['input_ids'] = tokenized.input_ids
    examples['attention_mask'] = tokenized.attention_mask
    return examples

if len(predict_csv_files) > 0:
    file2dataset = {f:load_dataset('csv', data_files=[os.path.join(data_base_dir, f)], delimiter=',', cache_dir=cache_dir)['train'] for f in predict_csv_files}
else:
    file2dataset = {f'{cfg.dataset_name}': load_dataset(cfg.dataset_name, cache_dir=cache_dir)['validation']}

for f,d in file2dataset.items():
    if cfg.column_renaming:
        for old_c, new_c in cfg.column_renaming.items():
            file2dataset[f] = file2dataset[f].rename_column(old_c, new_c)
    file2dataset[f] = get_inputs_and_targets(file2dataset[f], use_context=cfg.use_context, column_dict=column2name, num_proc=num_workers)
    file2dataset[f] = file2dataset[f].map(tokenize, batched=True, num_proc=num_workers)   
postproc = qa_postproc_answer

###########################################
########## MODEL  #########################
############################################
if use_checkpoint:
    model = pl_modules.QAModel.load_from_checkpoint(use_checkpoint,cfg=cfg).eval()
else:
    model = pl_modules.QAModel(cfg=cfg)
if torch.cuda.is_available:
    model.cuda()
    
###########################################
########## PREDICT  #########################
############################################
from src.predict.utils import generate_and_decode
listify = lambda x: ast.literal_eval(x) if '[' in x and ']' in x else x
for f, d in file2dataset.items():
    print(f"Predicting on {f}....")
    d = d.map(tokenize, batched=True, num_proc=num_workers)
    dl = DataLoader(d, batch_size=cfg.batch_size, shuffle=False)
    predictions = []
    targets = []
    for i, batch in tqdm(enumerate(dl)):
        input_ids = torch.cat([k.unsqueeze(0).t() for k in batch['input_ids']], dim=1)
        attn_mask = torch.cat([k.unsqueeze(0).t() for k in batch['attention_mask']], dim=1)
        preds =  generate_and_decode(model, input_ids=input_ids, 
                        attention_mask=attn_mask,batch_idx=i)
        predictions.extend([postproc(p) for p in preds])
        if type(batch[column2name['answer']]) != list:
             tags = [[normalize_squad(t_) for t_ in u] for u in [listify(t) for t in [batch[column2name['answer']]]]]
        else:
             tags = [[normalize_squad(t_) for t_ in u] for u in [listify(t) for t in batch[column2name['answer']]]]
        targets.extend(tags)
    ##save predictions
    if not cfg.dataset_name:
        df = pd.read_csv(os.path.join(data_base_dir, f)) ## place predictions
    else: 
        df = pd.DataFrame({})
    df[pred_column] = predictions
    os.makedirs(os.path.join(prediction_dir, cfg.model_name if not use_checkpoint else use_checkpoint), exist_ok=True)
    df.to_csv(os.path.join(prediction_dir, cfg.model_name if not use_checkpoint else use_checkpoint.replace('/','_'), f), index=False)
    
    ## compute metrics
    predictions = [normalize_squad(postproc(p)) for p in predictions]
    metrics = qa_metrics(targets, predictions)
    print("Results")
    print(metrics)