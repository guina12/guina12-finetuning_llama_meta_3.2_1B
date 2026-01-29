import os
import torch
from peft import PeftModel
from llama_cpp import Llama
from loaders.load_model import load_tokenizer
from transformers import AutoModelForCausalLM


#--------------------------------------------------------------------------------------------------------------------#

def  get_finetuned_model(output_dir):
    model = AutoModelForCausalLM.from_pretrained(output_dir, dtype = torch.float16)
    return model

#--------------------------------------------------------------------------------------------------------------------#

def merge_model(path, config, output_dir):
    if path is None:
        raise ValueError("Incorrect path")

    if os.path.isdir(output_dir) and os.path.isfile(os.path.join(output_dir, "config.json")):
        return get_finetuned_model(output_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        config["MODEL_NAME"],
        dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, path)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = load_tokenizer(config["MODEL_NAME"])
    tokenizer.save_pretrained(output_dir)

    return model

#--------------------------------------------------------------------------------------------------------------------------------#

def get_llama_cpp_q4_model(path):
     if path is None:
          raise ("Incorrect path")
     
     llm = Llama(
          model_path =  path,
          n_ctx = 8192,
          verbose =  False,
          n_threads =  3,
          n_gpu_layers = 8
     )

     return llm
     