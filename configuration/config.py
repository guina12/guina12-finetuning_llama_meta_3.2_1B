
#--------------------------------------------------------------------------------------------------------------------#

CONFIGURATIONS = {
    "MODEL_NAME":"meta-llama/Llama-3.2-1B-Instruct",
    "BATCH_SIZE_TRAIN":4,
    "BATCH_SIZE_EVAL":4,
    "GRADIENT_ACCUMULATION_STEPS" : 2,
    "LEARNING_RATE" : 3e-4,
    "MAX_LENGTH":64,
    "COLLATOR_FN":None,
    "STEPS":1000,
    "EVAL_STEPS":3000,
    "TEMPERATURE":0.1,
    "MAX_NEW_TOKENS": 500,
    "ASSISTANT_MODEL_NAME":"Qwen/Qwen2.5-0.5B",
    "PACKING":True,
    "PERCENT":0.8,
    "EPOCHS":1
}

#----------------------------------------------------------------------------------------------------------------------#

SYSTEM_PROMPT = (
    "You are an educational medical assistant. "
    "Provide clear, objective answers based on medical knowledge. "
    "Do not provide definitive diagnoses or treatment prescriptions."
)

#-----------------------------------------------------------------------------------------------------------------------#


def get_config(): return CONFIGURATIONS

#------------------------------------------------------------------------------------------------------------------------#


def get_parameter_config(parameter) :  return CONFIGURATIONS[parameter] 


#-------------------------------------------------------------------------------------------------------------------------#

def get_system_prompt(): return SYSTEM_PROMPT