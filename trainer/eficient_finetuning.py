from peft import get_peft_model,LoraConfig, prepare_model_for_kbit_training
from preprocess.preprocess_tokenizer import  apply_changes
from loaders.load_model import get_model , load_tokenizer
import torch

class Lora:
    def __init__(self):
        self.config_lora  = {
          "r" :8,
          "lora_alpha":16,
          "lora_dropout":0.05,
          "bias":"none",
          "task_type":"CAUSAL_LM"
        }
    
    def lora_init(self):
        lora = LoraConfig(
            r  = self.config_lora['r'],
            lora_alpha =  self.config_lora['lora_alpha'] ,
            lora_dropout =  self.config_lora['lora_dropout'],
            bias =  self.config_lora['bias'],
            task_type = self.config_lora['task_type']
        )
        return lora

class PrepareModel:
    def __init__(self, type, model_name):
        self.lora_init = Lora().lora_init()
        self.type = type
        self.model_name = model_name
        self.tokenizer = load_tokenizer(model_name) 
        self.model = get_model(type)
        
    
    def get_peft_model(self):
        self.model , self.tokenizer = apply_changes(self.model, self.tokenizer, self.model_name)
        self.model_q_type = prepare_model_for_kbit_training(
            self.model
        )
        self.prepared_model = get_peft_model(
            self.model_q_type, self.lora_init
        )
    
        return self.prepared_model, self.tokenizer