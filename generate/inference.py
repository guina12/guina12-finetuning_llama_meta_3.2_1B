import torch
torch._dynamo.disable()
from loaders.load_finetuned_model import get_finetuned_model, get_llama_cpp_q4_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from configuration.config import get_system_prompt


#-----------------------------------------------------------------------------------------------------------------------#

class Inference:
    def __init__(
        self,
        tokenizer,
        path,
        config,
        assistant_model = False,
        kv_cache = False,
    ):
        self.tokenizer = tokenizer
        self.path = path
        self.config = config
        self.system_prompt = get_system_prompt()
        self.assistant_model = assistant_model
        self.path_llama_cpp = 'models/llama_cpp_model/my_model_q4_k_m.gguf'
        self.kv_cache = kv_cache
        self.model = get_finetuned_model(self.path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def llama_cpp_inference_q4(self, message):
        prompt = self.gen_prompt(message)
        llm = get_llama_cpp_q4_model(self.path_llama_cpp)
        model = llm(
            prompt,
            max_tokens =  self.config["MAX_NEW_TOKENS"],
            stop = [self.tokenizer.eos_token],
            echo =  False
        )

        return model["choices"][0]["text"].strip()


    def assistant_model_gen(self):
        assistant_model = AutoModelForCausalLM.from_pretrained(
            self.config["ASSISTANT_MODEL_NAME"]
        )
        assistant_tokenizer = AutoTokenizer.from_pretrained(
            self.config["ASSISTANT_MODEL_NAME"]
        )
        return assistant_model, assistant_tokenizer

    def kv_cache_static(self):
        self.model.generation_config.cache_implementation = "static"
        return self.model

    def gen_prompt(self, message):
        prompt = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": message
            }
        ]

        formated_message = self.tokenizer.apply_chat_template(
            conversation = prompt,
            tokenize = False,
            add_generation_prompt = True
        )

        return formated_message

    def generate(self, message, config):
        formated_prompt = self.gen_prompt(message)

        tokens_ids = self.tokenizer(
            formated_prompt,
            add_special_tokens = False,
            return_tensors="pt"
        ).to(self.device)

        self.model = self.model.to(self.device)

        if self.kv_cache:
            self.model = self.kv_cache_static()

        if self.assistant_model:
            assistent_model, assistent_tokenizer = self.assistant_model_gen()
            response = self.model.generate(
                **tokens_ids,
                assistent_model = assistent_model,
                assistent_tokenizer = assistent_tokenizer,
                tokenizer = self.tokenizer,
                max_new_tokens = self.config["MAX_NEW_TOKENS"],
                temperature = self.config["TEMPERATURE"],
                do_sample = False,
                eos_token_id = self.tokenizer.eos_token_id
            )
               
        else:
            response =  self.model.generate(
                **tokens_ids,
                max_new_tokens = self.config["MAX_NEW_TOKENS"],
                temperature = self.config["TEMPERATURE"],
                do_sample = False,
                eos_token_id = self.tokenizer.eos_token_id
            )
           
        input_len = tokens_ids["input_ids"].shape[1]
        response = response[:, input_len:]

        text = self.tokenizer.batch_decode(
            response,
            skip_special_tokens = True
        )[0]
    
        return text