from loaders.data_loader import load_data
from datasets import Dataset
from loaders.load_model import load_tokenizer


#-----------------------------------------------------------------------------------------------------------------------#

def modity_tokenizer(
    tokenizer, 
    model_name,
    alternative_unk_token = "<|unk_token|>",
    special_tokens = None,
    tokens = None
):
    tokenizer = load_tokenizer(model_name)
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token

    if  pad_token is None  and unk_token is None:
        pad_token = alternative_unk_token
        unk_token = alternative_unk_token

    special_tokens_dict = {
        "pad_token" : pad_token , "unk_token": unk_token
    }

    if  special_tokens is not None:
        if isinstance(special_tokens, list):
            special_tokens_dict.update({'additional_special_tokens': special_tokens})

    tokenizer.add_special_tokens(special_tokens_dict)

    if  tokens is not None:
        if isinstance(tokens, list):
            tokenizer.add_tokens = tokens
    
    return tokenizer

#-----------------------------------------------------------------------------------------------------------------------#

def jinja_template(tokenizer):
  return ("{% for message in messages %}"
        f"{{{{'{tokenizer.bos_token}' + message['role'] + '\n' \
        + message['content'] + '{tokenizer.eos_token}' + '\n'}}}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        f"{{{{ '{tokenizer.bos_token}assistant\n' }}}}"
        "{% endif %}")

#-----------------------------------------------------------------------------------------------------------------------#


def add_chat_template(tokenizer, chat_template = None):
    if chat_template is None:
        tokenizer.chat_template = jinja_template(tokenizer)

    return tokenizer

#-----------------------------------------------------------------------------------------------------------------------#

def get_multiple_of(vocab_size):
   return 2**(bin(vocab_size)[::-1].find('1'))

#-------------------------------------------------------------------------------------------------------------------------#
 
def modify_model(model, tokenizer):
    '''
    if len(tokenizer) > model.config.vocab_size:
        pad_to_multiple_of = get_multiple_of(model.vocab_size)
        model.resize_token_embeddings(
            len(tokenizer), pad_to_multiple_of = pad_to_multiple_of
        )

    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.unk_token_id = tokenizer.unk_token_id

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.unk_token_id = tokenizer.unk_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
        '''
    
    return  model , tokenizer

#-------------------------------------------------------------------------------------------------------------------------#

def apply_changes(model , tokenizer, model_name):
    #tokenizer = modity_tokenizer(tokenizer, model_name)
    #tokenizer = add_chat_template(tokenizer, None)
    model, tokenizer = modify_model(model, tokenizer)

    return model , tokenizer

#-------------------------------------------------------------------------------------------------------------------------#
    