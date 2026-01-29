import os
import csv
import torch
import logging
from lm_eval import evaluator
from loaders.load_finetuned_model import merge_model

#---------------------------------------------------------------------------------------------------------------------------#

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("BenchmarkEvaluator")

#---------------------------------------------------------------------------------------------------------------------------#

class BenchmarkEvaluator:
    def __init__(self, lora_path, base_model_name, config):
        logger.info("Initializing benchmark evaluator")
        self.tasks = ["squad_completion","medmcqa","medqa_4options"]
        self.lora_path = lora_path
        self.base_model_name = base_model_name
        self.config = config
        self.output_dir = "models/model_merged/finetuned_hf"

        logger.info("Merging LoRA weights into base model")
        self.model = merge_model(self.lora_path, self.config, self.output_dir)
        logger.info("Model merge completed")

    #-----------------------------------------------------------------------------------------------------------------------#

    def evaluate(self):
        logger.info(f"Starting evaluation on tasks: {', '.join(self.tasks)}")

        results = evaluator.simple_evaluate(
            model = "hf",
            model_args = f"pretrained={self.output_dir}",
            tasks  = self.tasks,
            device = "cuda",
            batch_size = 1
        )

        logger.info("Evaluation finished")

        for task in self.tasks:
            if task in results["results"]:
                logger.info(f"Results for {task}: {results['results'][task]}")
        
        self.save_results_csv(results)

        return results

    #-----------------------------------------------------------------------------------------------------------------------#

    def save_results_csv(self, results, output_dir = "evaluation/metrics_evaluation/benchmark"):
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "benchmark_results.csv")

        with open(file_path, "w", newline = "", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["task", "metric", "value"])

            for task, metrics in results["results"].items():
                for metric, value in metrics.items():
                    writer.writerow([task, metric, value])

        logger.info(f"Results saved to {file_path}")
