from datasets import load_dataset

#--------------------------------------------------------------------------------#

def load_data():
    ds = load_dataset(
        'bio-nlp-umass/bioinstruct',
        split = "train"
    )

    return ds
