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


class CustomEvalCallback(Callback):
   def on_train_epoch_end(self, trainer, pl_module):
        from predict.qa_utils import qa_metrics, normalize_squad
        from src.predict.utils import generate_and_decode
        from src.data.postprocessing import qa_postproc_answer
        
        listify = lambda x: ast.literal_eval(x)
        local_predictions = []
        local_targets = []
        
        dl = trainer.datamodule.val_dataloader()
        pl_module.eval()
        
        for i, batch in enumerate(dl):  
            preds = generate_and_decode(pl_module, input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], batch_idx=i)
            local_predictions.extend(preds)
            tags = [[normalize_squad(t_) for t_ in u] for u in [listify(t) for t in batch['answers']]]
            local_targets.extend(tags)
            if pl_module.cfg.debug:
                print(f"targets: {tags}")
                print(f"preds: {preds}")

        # Convert lists to tensors
        local_predictions_tensor = torch.tensor(local_predictions).to(pl_module.device)
        local_targets_tensor = torch.tensor(local_targets).to(pl_module.device)
        
        # Gather predictions from all processes
        global_predictions = [torch.zeros_like(local_predictions_tensor) for _ in range(trainer.world_size)]
        global_targets = [torch.zeros_like(local_targets_tensor) for _ in range(trainer.world_size)]
        dist.gather(local_predictions_tensor, gather_list=global_predictions, dst=0)
        dist.gather(local_targets_tensor, gather_list=global_targets, dst=0)
        
        if pl_module.global_rank == 0:
            # Flatten lists
            global_predictions = [item for sublist in global_predictions for item in sublist]
            global_targets = [item for sublist in global_targets for item in sublist]
            
            postproc = qa_postproc_answer if not pl_module.cfg.disentqa else disent_qa_postproc_answer
            global_predictions = [normalize_squad(postproc(p)) for p in global_predictions]
            metrics = qa_metrics(global_targets, global_predictions)
            
            pl_module.log_dict({f"eval_{k}": v for k, v in metrics.items()})

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

