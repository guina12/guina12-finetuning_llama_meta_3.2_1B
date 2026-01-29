import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

#------------------------------------------------------------------------------------------------------------------------------------------------------#

from configuration.config import get_config
from evaluation.judge_evaluation import judge_eval
from evaluation.reference_evaluation import bert_metric
from evaluation.benchmark_evaluation import BenchmarkEvaluator


#------------------------------------------------------------------------------------------------------------------------------------------------------#

config = get_config()
MODEL_NAME = config["MODEL_NAME"]
PATH = "adapters/medcare_ia_adapter"
MAX_LINES = 50

#----------------------------------------BERT EVALUATION-----------------------------------------------------------------------------------------------#
'''
bert_metric(n_iteration = MAX_LINES, num_eval_dataset =  5)
'''

#----------------------------------------JUDGE EVALUATION----------------------------------------------------------------------------------------------#
''''
judge_eval('fatualidade', MAX_LINE_EVAL = MAX_LINES)
'''


#---------------------------------------BENCHAMRK EVALUATION------------------------------------------------------------------------------------------#

bench = BenchmarkEvaluator(lora_path = PATH, base_model_name = MODEL_NAME, config = config)
bench.evaluate()


#------------------------------------------------------------------------------------------------------------------------------------------------------#