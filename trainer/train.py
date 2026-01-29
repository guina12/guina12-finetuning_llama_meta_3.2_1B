from finetuning_medcare.trainer.eficient_finetuning import PrepareModel
from finetuning_medcare.configuration.config import get_config
from finetuning_medcare.preprocess.preprocess_data import get_data
from trl  import SFTTrainer, SFTConfig

#---------------------------------------------------------------------------------------#

class Trainer:
    def __init__(self, type , config):
        self.type  = type
        self.prepared_model = PrepareModel(type , config["MODEL_NAME"])
        self.model , self.tokenizer = self.prepared_model.get_peft_model()
        self.train_data, self.eval_data = get_data(config["PERCENT"], config["MODEL_NAME"])
        self.config_arguments = SFTConfig(
            #Dataset
            output_dir = "./medcare_ia",
            packing = config["PACKING"],
            max_length = config["MAX_LENGTH"],
            fp16 = False,
            bf16 =  True,
            dataloader_num_workers = 4,   
            dataloader_pin_memory = True,
            optim = "adamw_torch_fused",
            #Gradients / Memory
            gradient_accumulation_steps = config["GRADIENT_ACCUMULATION_STEPS"],
            gradient_checkpointing =  False,
            auto_find_batch_size =  False,
            gradient_checkpointing_kwargs = {"use_reentrant": False},
            per_device_train_batch_size = config["BATCH_SIZE_TRAIN"],
            per_device_eval_batch_size  = config["BATCH_SIZE_EVAL"],
            #Training
            num_train_epochs =  config["EPOCHS"],
            learning_rate = config["LEARNING_RATE"],
            #Env and Logging
            report_to ='tensorboard',
            logging_dir='./logs',
            logging_strategy='steps',
            logging_steps = config["STEPS"],
            eval_strategy =  "steps",
            eval_steps = config["EVAL_STEPS"],
            save_strategy = 'steps',
            save_steps = config["STEPS"],
        )

        self.trainer = SFTTrainer(
            self.model,
            processing_class =  self.tokenizer,
            train_dataset =  self.train_data,
            eval_dataset  =  self.eval_data,
            data_collator =  config["COLLATOR_FN"],
            args =  self.config_arguments
        )
    
    def start_trainer(self):
        self.trainer.train()
