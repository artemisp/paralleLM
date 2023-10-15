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

def generate(pl_module, input_ids, attention_mask, batch_idx=None):
    model = pl_module.model
    input_ids = input_ids.type_as(next(model.parameters())).long()
    attention_mask=attention_mask.type_as(next(model.parameters()))
    with torch.no_grad():   
        pred_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=pl_module.cfg.output_max_length,
                                    repetition_penalty=pl_module.cfg.repetition_penalty, length_penalty=pl_module.cfg.length_penalty,
                                    no_repeat_ngram_size=pl_module.cfg.no_repeat_ngram_size, num_beams=pl_module.cfg.num_beams,
                                    early_stopping=pl_module.cfg.early_stopping, min_new_tokens=pl_module.cfg.min_new_tokens)
    return pred_ids

def generate_and_pad(pl_module, input_ids, attention_mask, batch_idx=None, pad_to_length=None):
    # Generate sequences using the provided method
    pred_ids = generate(pl_module, input_ids, attention_mask, batch_idx)

    # Pad the sequences if a pad_to_length is provided
    if pad_to_length:
        # Assuming PAD token id is 0. Adjust if it's different for your tokenizer
        PAD_TOKEN_ID = 0
        curr_length = pred_ids.shape[1]
        
        # If the current sequence length is less than the pad_to_length, pad it
        if curr_length < pad_to_length:
            padding = torch.full((pred_ids.shape[0], pad_to_length - curr_length), PAD_TOKEN_ID, dtype=torch.long, device=pred_ids.device)
            pred_ids = torch.cat([pred_ids, padding], dim=1)
        elif curr_length > pad_to_length:
            pred_ids = pred_ids[:, :pad_to_length]

    return pred_ids
