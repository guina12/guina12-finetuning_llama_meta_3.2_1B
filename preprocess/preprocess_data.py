from loaders.load_model import load_tokenizer
from loaders.data_loader import load_data
from datasets import Dataset
import logging

#-------------------------------------------------------------------------------------------------------#

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an educational medical assistant. "
    "Provide clear, objective answers based on medical knowledge. "
    "Do not provide definitive diagnoses or treatment prescriptions."
)

#-------------------------------------------------------------------------------------------------------#

def get_data(percent, model_name):
    logger.info("Starting get_data()")

    dataset = format_data(model_name)

    logger.info(f"Formatted dataset size: {len(dataset['text'])}")

    ratio = int(len(dataset['text']) * percent)
    logger.info(f"Train size: {ratio} | Eval size: {len(dataset['text']) - ratio}")

    train_data = Dataset.from_dict({"text": dataset["text"][:ratio]})
    eval_data  = Dataset.from_dict({"text": dataset["text"][ratio:]})

    logger.info("get_data() finished")

    return train_data, eval_data

#--------------------------------------------------------------------------------------------------------#

def format_data(model_name):
    logger.info("Loading raw dataset...")
    data = load_data()

    logger.info(f"Total samples loaded: {len(data['instruction'])}")

    logger.info("Raw data sample:")
    logger.info(f"instruction: {data['instruction'][0][:200]}")
    logger.info(f"input: {data['input'][0][:200]}")
    logger.info(f"output: {data['output'][0][:200]}")

    output_txt = []

    for i in range(len(data['instruction'])):
        f_data = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": data['instruction'][i] + data['input'][i]},
            {"role": "assistant", "content": data["output"][i]},
        ]
        output_txt.append(f_data)

        if i == 0:
            logger.info("First conversation (messages format):")
            logger.info(f_data)

        if i % 5000 == 0 and i > 0:
            logger.info(f"{i} samples processed...")

    dataset = Dataset.from_dict({"messages": output_txt})

    logger.info("Applying chat template...")
    dataset = format_dataset(dataset, model_name)

    logger.info("format_data() finished")

    return dataset

#------------------------------------------------------------------------------------------------------#

def format_dataset(dataset, model_name):
    messages = dataset["messages"]
    output_txt = []

    tokenizer = load_tokenizer(model_name)
    logger.info("Tokenizer loaded")

    for idx, ms in enumerate(messages):
        formatted_conversation = tokenizer.apply_chat_template(
            conversation=ms,
            tokenize=False,
            add_generation_prompt=False
        )
        output_txt.append(formatted_conversation)

        if idx == 0:
            logger.info("First formatted text (chat template output):")
            logger.info(formatted_conversation[:1000])

        if idx % 5000 == 0 and idx > 0:
            logger.info(f"{idx} texts formatted...")

    logger.info("Chat template applied to all samples")

    return {
        "text": output_txt
    }

#-----------------------------------------------------------------------------------------------------------------------------#