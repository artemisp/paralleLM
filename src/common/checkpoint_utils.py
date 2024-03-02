def trim_prefix(state_dict):
    state_dict = {k:v for k, v in state_dict.items() if 'learned_embedding' in k}
    return state_dict

def trim_lora(state_dict):
    state_dict = {k:v for k, v in state_dict.items() if 'lora' in k.lower()}
    return state_dict
