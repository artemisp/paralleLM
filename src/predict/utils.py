import torch

def generate_and_decode(pl_module, input_ids, attention_mask, batch_idx=None):
    model = pl_module.model
    input_ids = input_ids.type_as(next(model.parameters())).long()
    attention_mask=attention_mask.type_as(next(model.parameters()))
    with torch.no_grad():   
        pred_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=pl_module.cfg.output_max_length,
                                    repetition_penalty=pl_module.cfg.repetition_penalty, length_penalty=pl_module.cfg.length_penalty,
                                    no_repeat_ngram_size=pl_module.cfg.no_repeat_ngram_size, num_beams=pl_module.cfg.num_beams,
                                    early_stopping=pl_module.cfg.early_stopping)
    decoded_preds = [pl_module.tokenizer.decode(pred_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                pred_id in pred_ids]
    return decoded_preds