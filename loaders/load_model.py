from transformers import AutoModelForCausalLM, AutoTokenizer
from quantization.quantization_config import quantization_type
from configuration.config import get_parameter_config
import torch 

#-------------------------------------------------------------------------------------#

def get_model(type):
    model_name = get_parameter_config("MODEL_NAME")
    quantization_config = quantization_type(type)

    if quantization_config is None:
        return ValueError("Error, Failed to load the quantization type")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = quantization_config,
        attn_implementation = "sdpa",
        device_map = "cuda:0",
        dtype = torch.float32
    )
    return  model

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    return tokenizer