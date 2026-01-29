import numpy as np
from datasets import Dataset
from loaders.data_loader import load_data

MAX_LINES_PER_EVAL_DATASET = 1000
MAX_LINES_EVAL = 5000

#------------------------------------------------------------------------------------------------------------------------------#
 
def get_data():
    instructions , outputs = [], []
    dataset = load_data()

    for i in range(MAX_LINES_EVAL):
        instruction = dataset["instruction"][i] + dataset["input"][i]
        output = dataset["output"][i]
        instructions.append(instruction)
        outputs.append(output)

    eval_dataset = Dataset.from_dict({
        "instruction": instructions,
        "output": outputs
    })

    return eval_dataset

#------------------------------------------------------------------------------------------------------------------------------#

class StackDataset:
    def __init__(self, capacity):
        self.capacity = capacity
        self.lastIndex = -1 
        self.valuers = np.empty(self.capacity, dtype =  object)

    def push(self, dataset):
        if self.lastIndex == self.capacity - 1 :
            raise IndexError()
        else:
            self.lastIndex +=1
            self.valuers[self.lastIndex] = dataset

#------------------------------------------------------------------------------------------------------------------------------#


def split_data(num_eval_datasets):
    stack = StackDataset(num_eval_datasets)
    eval_dataset = get_data()

    for i in range(num_eval_datasets):
        start = i * MAX_LINES_PER_EVAL_DATASET
        end = (i + 1) * MAX_LINES_PER_EVAL_DATASET
        block = Dataset.from_dict({
            "instruction": eval_dataset["instruction"][start:end],
            "output": eval_dataset["output"][start:end]
        })

        stack.push(block)
    
    return stack