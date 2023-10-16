from pytorch_lightning import Callback,LightningModule
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
from dotenv import load_dotenv
import ast
import sys
import ast
import torch.distributed as dist
sys.path.append(os.getcwd())


# Load the variables from the .env file
load_dotenv(os.getcwd()+'/.env')
# Get the data root dir
cache_dir = os.getenv('CACHE_DIR')
class QAModel(LightningModule):
    """
        A PyTorch Lightning module that encapsulates a Hugging Face transformer model
        for Seq2Seq tasks like machine translation or text summarization.

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
    def __init__(self,cfg, learning_rate=None, tokenizer=None):
        """
        Constructs all the necessary attributes for the QAModel object.

        Parameters
        ----------
        cfg : object
            The configuration object with model and training parameters.
        learning_rate : float, optional
            The learning rate for the optimizer.
        tokenizer : transformers.AutoTokenizer, optional
            The tokenizer associated with the transformer model.
        checkpoint : str, optional
            The path to a checkpoint to load a pre-trained model.
        """
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, cache_dir=cache_dir, return_dict=True)
        self.tokenizer = tokenizer if tokenizer != None else AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cache_dir)
        # for auto
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = cfg.learning_rate
        self.cfg = cfg
        self.batch_size = cfg.batch_size

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits
    
    def on_train_epoch_end(self):
        predictions = []
        targets = []
        dl = self.trainer.datamodule.val_dataloader()
        self.eval()
        for i,batch in enumerate(dl):  
                preds = generate_and_decode(self, input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'],batch_idx=i)
                predictions.extend(preds)
                tags = [[normalize_squad(t_) for t_ in u] for u in [listify(t) for t in batch['answers']]]
                targets.extend(tags)
                if self.cfg.debug:
                    print(f"targets: {tags}")
                    print(f"preds: {preds}")
        postproc = qa_postproc_answer if not self.cfg.disentqa else disent_qa_postproc_answer
        predictions = [normalize_squad(postproc(p)) for p in predictions]
        metrics = qa_metrics(targets, predictions)
        self.log_dict({f"eval_{k}": v for k,v in metrics.items()}, sync_dist=True)
        self.train()
        
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.cfg.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.cfg.optimizer == 'AdamW':
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        if self.cfg.optimizer == 'Adafactor':
            from transformers import Adafactor
            return Adafactor(self.parameters(), lr=self.learning_rate, relative_step=False)

