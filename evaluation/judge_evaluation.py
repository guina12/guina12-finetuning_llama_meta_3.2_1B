import os
import  json
import logging
import pandas as pd
from evaluation.slice import get_data
from configuration.config import get_config
from generate.inference import Inference
from loaders.load_model import load_tokenizer
from evaluation.prompts.prompt import prompt_instruction
from evaluation.openai_api.openai_api_client import OpenAIApiClient

# -----------------------------------------------------------------------------------------------------------------------------#

# Logger config (GLOBAL, simples e limpa)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------------------------------------------------#

PATH = "adapters/medcare_ia_adapter"

#-------------------------------------------------------------------------------------------------------------------------#

def load_llama_model_q4_inference():
    logger.info("Loading LLaMA Q4 model for inference")
    config = get_config()
    tokenizer = load_tokenizer(config['MODEL_NAME'])
    inference = Inference(tokenizer, PATH, config, False, False)
    logger.info("LLaMA model loaded successfully")
    return inference

#-------------------------------------------------------------------------------------------------------------------------#

def load_openai_api():
    logger.info("Initializing OpenAI API client")
    chat_openai = OpenAIApiClient()
    logger.info("OpenAI API client initialized")
    return chat_openai

#-------------------------------------------------------------------------------------------------------------------------#

def judge_eval(criteria, MAX_LINE_EVAL):
    logger.info("Starting judge evaluation")
    output_response_api = []

    logger.info("Loading evaluation dataset")
    eval_dataset = get_data()

    llama_model = load_llama_model_q4_inference()
    openai_client = load_openai_api()

    for lines in range(MAX_LINE_EVAL):
        logger.info(f"Evaluating line {lines + 1}/{MAX_LINE_EVAL}")

        user_question = eval_dataset["instruction"][lines]
        logger.debug(f"User question: {user_question}")

        model_response = llama_model.llama_cpp_inference_q4(user_question)
        logger.debug(f"Model response: {model_response}")

        prompt = prompt_instruction(criteria, user_question, model_response)
        logger.debug("Prompt constructed successfully")

        response_api = openai_client.generate(prompt)
        logger.info("Received response from OpenAI API")

        output_response_api.append(response_api)

    logger.info("Judge evaluation completed successfully")
    response = judge_structured_outputs(output_response_api)
    logger.info("Calculating factuality percentage for the evaluation dataset...")
    compute_factuality(response)
    return output_response_api

# -------------------------------------------------------------------------------------------------------------------------#


def compute_factuality(response, output_dir="evaluation/metrics_evaluation/judge_eval", filename="factuality_results.csv"):
    os.makedirs(output_dir, exist_ok=True)

    factuality_percentage = (response["score_acc"].mean() / 3) * 100
    strict_factuality = (response["score_acc"] == 3).mean() * 100

    print(f"Average factuality: {factuality_percentage:.1f}%")
    print(f"Strict factuality (score=3): {strict_factuality:.1f}%")

    output_path = os.path.join(output_dir, filename)
    response.to_csv(output_path, index=False)

    return {
        "factuality_percentage": factuality_percentage,
        "strict_factuality": strict_factuality,
        "csv_path": output_path
    }


def judge_structured_outputs(response):
    data = {
        "analysis_acc":[],
        "score_acc":[],
        "analysis_style":[],
        "score_style":[]
    }  
    for i in range(len(response)):
        score = json.loads(response[i])
        data['analysis_acc'].append(score['accuracy']['analysis'])
        data['score_acc'].append(score['accuracy']['score'])
        data['analysis_style'].append(score['style']['analysis'])
        data['score_style'].append(score['style']['score'])

        dataframe  = pd.DataFrame(data)

    return  dataframe

