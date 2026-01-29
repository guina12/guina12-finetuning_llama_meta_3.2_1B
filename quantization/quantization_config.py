from transformers  import BitsAndBytesConfig
import torch

QUANTIZATION_TYPE =  ["4bit","8bit"]

def quantization_type(type):
    if type in QUANTIZATION_TYPE:
        if type == "4bit":
            bnb_config_4bit =  BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_compute_dtype = torch.bfloat16,
                bnb_4bit_quant_type = "nf4"
            )
            return bnb_config_4bit
        
        elif  type == '8bit': 
            bnb_config_8bit =  BitsAndBytesConfig(
            load_in_8bit = True,
            llm_int8_threshold =  6.0
        )
        return bnb_config_8bit
    else:
        return None


    