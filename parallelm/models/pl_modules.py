import pytorch_lightning as pl
import torch
import transformers 


import os
import sys

from torch import nn
from tqdm import tqdm
from dotenv import load_dotenv

from parallelm.models.soft_embedding import SoftEmbedding
from parallelm.common.checkpoint_utils import trim_lora, trim_prefix
from transformers import BitsAndBytesConfig, AutoTokenizer, LlamaTokenizer, T5Config, AutoModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import parallelm.data.postprocessing as postprocessing

# Load the variables from the .env file
load_dotenv(os.getcwd()+'/.env')
# Get the data root dir
cache_dir = os.getenv('CACHE_DIR', "./.cache")
HF_TOKEN = os.getenv('HF_ACCESS_TOKEN', None)

class CustomModule(pl.LightningModule):
    def __init__(self, learning_rate=None, tokenizer=None, predict=False, **kwargs):
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

        super().__init__()
        
        self.kwargs =  kwargs
        self.debug = self.kwargs.get('debug', False)

        
        # for auto tuning
        self.learning_rate = learning_rate if learning_rate is not None else self.kwargs.get('optimizer_config',{}).get('lr', 1e-4)
        self.optimizer_type = self.kwargs.get('optimizer_type', 'AdamW')
        self.optimizer_config = self.kwargs.get('optimizer_config',{})
        self.optimizer_config['lr'] = self.learning_rate
        self.generation_kwargs = self.kwargs.get('generation_kwargs', {})
        self.auto_model_class = self.kwargs.get('auto_model_class', 'AutoModelForSeq2SeqLM')
        self.tokenizer = tokenizer
        self.lora = self.kwargs.get('lora', False)
        self.predict = predict
        self.soft_prompt = self.kwargs.get('prefix_tuning', False)
        
    def setup(self, stage):
        super().setup(stage)
        if self.tokenizer == None:
            self.tokenizer = self.kwargs.get('tokenization_kwargs', {}).get('tokenizer_name', 't5-small')
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer,
                                                        cache_dir=cache_dir, 
                                                        token=HF_TOKEN,
                                                        **self.kwargs.get('tokenization_kwargs', {})) if isinstance(self.tokenizer, str) else self.tokenizer 
            self.tokenizer.pad_token = getattr(self.tokenizer,self.kwargs.get('tokenization_kwargs', {}).get("pad_token","pad_token"))
            self.tokenizer.pad_token_id = getattr(self.tokenizer,f'{self.kwargs.get("tokenization_kwargs", {}).get("pad_token","pad_token")}_id')
        
        
        auto_model_class = getattr(transformers, self.auto_model_class)
        if self.kwargs.get('keep_in_fp32_modules', None):
            auto_model_class._keep_in_fp32_modules= self.kwargs['keep_in_fp32_modules']
        
        
        quantization_config = self.kwargs.get('quantization_config', {})
        quantization_precision = str(self.kwargs.get('quantization_precision', '4'))
        if quantization_config.get(f'bnb_{quantization_precision}bit_compute_dtype', None) and isinstance( quantization_config.get(f'bnb_{quantization_precision}bit_compute_dtype', None), str):
            quantization_config[f'bnb_{quantization_precision}bit_compute_dtype'] = getattr(torch, quantization_config[f'bnb_{quantization_precision}bit_compute_dtype'])
        self.model_name= self.kwargs.get('model_name', 't5-small')
        
        if torch.cuda.is_available():
            device_map= {"": torch.cuda.current_device()}
        else:
            device_map = "auto"
        # if self.device.index != None:
        #     device_map= {"": self.device.index}
        # else:
        #     device_map = "auto"
        # else:
        # device_map = "auto"
        # if torch.cuda.is_available():
        #     device_map= {"": self.device.index}
        self.model = auto_model_class.from_pretrained(self.model_name, 
                                                    cache_dir=cache_dir, 
                                                    token=HF_TOKEN,
                                                    return_dict=True, 
                                                    quantization_config=BitsAndBytesConfig(**quantization_config) if self.kwargs.get("quantization", False) else None,   
                                                    device_map = device_map,
                                                    trust_remote_code=True,
                                                    )
        if self.kwargs.get("quantization", False):
            print("Quantizing model...")
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=self.kwargs.get('gradient_checkpointing', False))

        if self.kwargs.get('lora', False):
            config = LoraConfig(
                **self.kwargs.get('lora_config', {}),
            )
            self.model = get_peft_model(self.model, config)

        if self.kwargs.get('freeze_encoder', False):
            layers_to_feeze = self.kwargs.get('freeze_encoder_layers', None)
            if layers_to_feeze == None:
                layers_to_feeze = list(range(len(self.model.encoder.layer)))
            for l in layers_to_feeze:
                # todo: t5 has "blocks" check name for model
                if 't5' in self.model_name:
                    for p in self.model.encoder.block[l].parameters():
                        p.requires_grad=False
                else:
                    for p in self.model.encoder.layer[l].parameters():
                        p.requires_grad=False
        if self.kwargs.get('freeze_decoder', False):
            layers_to_feeze = self.kwargs.get('freeze_decoder_layers', None)
            if layers_to_feeze == None:
                layers_to_feeze = list(range(len(self.model.decoder.layer)))
            for l in layers_to_feeze:
                for p in self.model.decoder.layer[l].parameters():
                    p.requires_grad=False
        
        self.soft_prompt = self.kwargs.get('prefix_tuning', False)
        self.soft_prompt_ntokens = self.kwargs.get('n_prefix_tokens', False)
        if self.soft_prompt:
            print("Freezing llm")
            for param in self.model.parameters():
                param.requires_grad = False
        if self.soft_prompt:
            print("Initializing prefix")
            initialize_from_vocab = self.kwargs.get('initialize_from_vocab', False)
            s_wte = SoftEmbedding(self.model.get_input_embeddings(),
                                n_tokens=self.soft_prompt_ntokens,
                                initialize_from_vocab=initialize_from_vocab
                                )
            if torch.cuda.is_available():
                s_wte.to(torch.cuda.current_device())
            self.model.set_input_embeddings(s_wte)
        
        
    def forward(self, input_ids, attention_mask, labels=None, output_hidden_states=False, output_attentions=False):
        if output_hidden_states:
            output = self.model(input_ids=input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask, labels=labels,  output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        else:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,  output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        return output

    def training_step(self, batch, batch_idx):
        indices, batch = batch
        if type(batch["input_ids"]) == list:
            batch["input_ids"] = torch.stack(batch["input_ids"]).t()
            batch["attention_mask"] = torch.stack(batch["attention_mask"]).t()
            batch["labels"] = torch.stack(batch["labels"]).t()
        
        if (len(batch['input_ids'].shape) == 3):
                batch['input_ids'] = batch['input_ids'].squeeze(1)
                batch['attention_mask'] = batch['attention_mask'].squeeze(1)
                batch["labels"] = batch["labels"].squeeze(1)
        output = self(batch["input_ids"], batch["attention_mask"],  batch["labels"])
        loss = output.loss
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True, batch_size=batch["input_ids"].shape[0])
        return loss
        
    def predict_step(self, batch, batch_idx):
        if type(batch["input_ids"]) == list:
            if type(batch["input_ids"][0]) == torch.Tensor:
                batch["input_ids"] = torch.stack(batch["input_ids"]).t()
                batch["attention_mask"] = torch.stack(batch["attention_mask"]).t()
            else:
                batch["input_ids"] = torch.tensor(batch["input_ids"]).to(self.device)
                batch["attention_mask"] = torch.tensor(batch["attention_mask"]).to(self.device)
        if 'decoder_input_ids' in batch:
            preds = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], decoder_input_ids=batch['decoder_input_ids'], **self.generation_kwargs)
        else:
            if (len(batch['input_ids'].shape) == 3 ):
                batch['input_ids'] = batch['input_ids'].squeeze(1)
                batch['attention_mask'] = batch['attention_mask'].squeeze(1)
            preds = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], **self.generation_kwargs)
        decoded_preds = []
        for i in range(len(preds)):
            decoded_preds.append(self.tokenizer.decode(preds[i][batch['input_ids'][i].shape[0]:], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        postproc = getattr(postprocessing, self.kwargs.get("postproc_fn", "identity"))
        decoded_preds = [postproc(p) for p in decoded_preds]
        if self.debug:
            print(decoded_preds)
        output = []
        for i in range(len(decoded_preds)):
            res = {'prediction': decoded_preds[i]}
            for k in batch:
                if not isinstance(batch[k], torch.Tensor):
                    res[k] = batch[k][i]
            output.append(res)
        return output

    def validation_step(self, batch, batch_idx):
        if type(batch["input_ids"]) == list:
            batch["input_ids"] = torch.stack(batch["input_ids"]).t()
            batch["attention_mask"] = torch.stack(batch["attention_mask"]).t()
            batch["labels"] = torch.stack(batch["labels"]).t()
        if (len(batch['input_ids'].shape) == 3):
                batch['input_ids'] = batch['input_ids'].squeeze(1)
                batch['attention_mask'] = batch['attention_mask'].squeeze(1)
                batch["labels"] = batch["labels"].squeeze(1)
        output = self(batch["input_ids"], batch["attention_mask"],  batch["labels"])
        loss = output.loss
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True, batch_size=batch["input_ids"].shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        if type(batch["input_ids"]) == list:
            batch["input_ids"] = torch.stack(batch["input_ids"]).t()
            batch["attention_mask"] = torch.stack(batch["attention_mask"]).t()
            batch["labels"] = torch.stack(batch["labels"]).t()
        if (len(batch['input_ids'].shape) == 3 ):
                batch['input_ids'] = batch['input_ids'].squeeze(1)
                batch['attention_mask'] = batch['attention_mask'].squeeze(1)
                batch["labels"] = batch["labels"].squeeze(1)
        output = self(batch["input_ids"], batch["attention_mask"],  batch["labels"])
        loss = output.loss
        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True, batch_size=batch["input_ids"].shape[0])
        return loss

    def configure_optimizers(self):
        if self.optimizer_config.get('scheduler', None):
            scheduler = self.optimizer_config['scheduler']
            scheduler_config = self.optimizer_config.get('scheduler_config', {})
            del self.optimizer_config['scheduler']
            if scheduler_config:
                del self.optimizer_config['scheduler_config']

        if self.optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(self.trainer.model.parameters(),**self.optimizer_config)
        if self.optimizer_type == 'AdamW':
            optimizer= torch.optim.AdamW(self.trainer.model.parameters(), **self.optimizer_config)
        if self.optimizer_type == 'Adafactor':
            from transformers import Adafactor
            optimizer = Adafactor(self.trainer.model.parameters(), **self.optimizer_config)
        if 'bnb' in self.optimizer_type:
            import bitsandbytes as bnb
            optimizer = getattr(bnb.optim, self.optimizer_type.split('.')[-1])(self.trainer.model.parameters(), **self.optimizer_config)
        if self.optimizer_config.get('scheduler', None):
            scheduler = getattr(torch.optim.lr_scheduler,scheduler)
            return [optimizer], [scheduler(optimizer, **scheduler_config)]
        else:
            return optimizer  
    
    
    def on_save_checkpoint(self, checkpoint):
        if self.lora:
            checkpoint['state_dict'] = trim_lora(checkpoint['state_dict'])
        elif self.soft_prompt:
            checkpoint['state_dict'] = trim_prefix(checkpoint['state_dict'])
        return checkpoint    
    
    def load_state_dict(self, state_dict, strict=True):
        if self.lora:
            self.model.load_state_dict({'.'.join(k.split('.')[1:]):v for k,v in state_dict.items()}, strict=False)
        elif self.soft_prompt:
            if 'model.model.embed_tokens.learned_embedding' in state_dict or strict:
                self.model.model.embed_tokens.learned_embedding.data = state_dict['model.model.embed_tokens.learned_embedding']