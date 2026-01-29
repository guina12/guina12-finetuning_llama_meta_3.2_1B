import os
import logging
import evaluate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generate.inference import Inference
from configuration.config import get_config
from evaluation.slice import split_data
from loaders.load_model import load_tokenizer


# -------------------------------------------------------------------------------------------------------------#

# LOGGING CONFIG
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------------------------------#

PATH = "adapters/medcare_ia_adapter"
METRICS = ["f1_score", "precision", "recall"]

# ----------------------------------------------------------------------------------------------------------------------#


def load_bert_score():
    logger.debug("Computing BERTScore...")
    bert_score_metric = evaluate.load("bertscore")
    return bert_score_metric


# ----------------------------------------------------------------------------------------------------------------------#

def load_llama_model_q4():
    logger.info("Loading LLaMA Q4 model...")
    config = get_config()
    tokenizer = load_tokenizer(config["MODEL_NAME"])
    inference = Inference(tokenizer, PATH, config, False, True)
    logger.info("Model loaded successfully.")
    return inference

# -----------------------------------------------------------------------------------------------------------------------#

def add_score(score, bert_score_per_dataset, it):
    logger.debug(f"Storing metrics for iteration {it}")
    bert_score_per_dataset['n_iteration'].append(it)
    bert_score_per_dataset['f1_score'].append(score['f1'][0])
    bert_score_per_dataset['recall'].append(score['recall'][0])
    bert_score_per_dataset['precision'].append(score['precision'][0])

# -------------------------------------------------------------------------------------------------------------------------#

def store_all_scores_per_dataset(n_iteration, num_eval_dataset):
    logger.info(
        f"Starting evaluation: {num_eval_dataset} datasets | {n_iteration} iterations"
    )
    score_per_dataset_eval = bert_evaluation(n_iteration, num_eval_dataset)

    results = {
        "dataset_A": score_per_dataset_eval[0],
        "dataset_B": score_per_dataset_eval[1],
        "dataset_C": score_per_dataset_eval[2],
        "dataset_D": score_per_dataset_eval[3],
        "dataset_E": score_per_dataset_eval[4],
    }

    logger.info("Evaluation completed for all datasets.")
    return results

# -------------------------------------------------------------------------------------------------------------------------#

def bert_evaluation(n_iteration, num_eval_dataset):
    logger.info("Splitting datasets...")
    all_score_per_dataset_eval = []

    stack_dataset = split_data(num_eval_dataset)
    llama_model = load_llama_model_q4()
    bert_metric = load_bert_score()

    for i in range(len(stack_dataset.valuers)):
        logger.info(f"Evaluating dataset {i + 1}/{len(stack_dataset.valuers)}")
        bert_score_per_dataset = {
            'n_iteration': [],
            'f1_score': [],
            'recall': [],
            'precision': []
        }
        dataset_i = stack_dataset.valuers[i]
        for it in range(n_iteration):
            idx = np.random.choice(len(dataset_i), 1, replace=False)
            msg = np.array(dataset_i['instruction'])[idx][0]
            reference = np.array(dataset_i["output"])[idx][0]

            out = llama_model.llama_cpp_inference_q4(msg)
            score = bert_metric.compute(
                predictions = [out],
                references = [reference],
                lang = 'en'
            )

            add_score(score, bert_score_per_dataset, it)

        all_score_per_dataset_eval.append(bert_score_per_dataset)

    return all_score_per_dataset_eval

# --------------------------------------------------------------------------------------------------------------------#

def bert_metric(n_iteration, num_eval_dataset):
    logger.info("Generating metric plots...")
    results = store_all_scores_per_dataset(n_iteration, num_eval_dataset)

    for metric in METRICS:
        logger.info(f"Plotting metric: {metric}")
        plt.figure()
        data = [results[k][metric] for k in results]
        plt.violinplot(data, showmeans=True)
        plt.xticks(range(1, len(results) + 1), results.keys())
        plt.ylabel(metric)
        plt.title(f"Dataset comparison — {metric} (Bootstrap)")
        plt.grid(True)
        plt.show()

    logger.info("Plots completed.")
    save_metrics(results)

#-------------------------------------------------------------------------------------------------------------------------------#

def save_metrics(results, output_dir="evaluation/metrics_evaluation/reference_eval"):
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, dataset_data in results.items():
        logger.info(f"Saving metrics for {dataset_name}...")

        df = pd.DataFrame({
            "iteration": dataset_data["n_iteration"],
            "f1_score": dataset_data["f1_score"],
            "precision": dataset_data["precision"],
            "recall": dataset_data["recall"],
        })

        file_path = os.path.join(output_dir, f"{dataset_name}_metrics.csv")
        df.to_csv(file_path, index=False)

        logger.info(f"Metrics saved at {file_path}")