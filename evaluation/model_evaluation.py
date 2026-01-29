import os
import json
import pandas as pd

#----------------------------------------------------------------------------------------------------------------------------#

def generate_csv_from_last_checkpoint_with_bits(
    base_dir,
    output_csv = "evaluation/metrics_evaluation/model/metrics_last_checkpoint.csv",
    tokens_per_word = 3,
    bits_per_char_factor = 7/8
):
    checkpoints = [
        d for d in os.listdir(base_dir)
        if d.startswith("checkpoint-")
    ]

    if not checkpoints:
        raise ValueError("No checkpoints found")

    last_checkpoint = max(checkpoints, key = lambda x: int(x.split("-")[1]))
    checkpoint_path = os.path.join(base_dir, last_checkpoint, "trainer_state.json")

    with open(checkpoint_path, "r", encoding = "utf-8") as f:
        state = json.load(f)

    rows = []

    for log in state["log_history"]:
        entropy = log.get("entropy") or log.get("eval_entropy")

        if entropy is not None:
            bpt = entropy
            bpc = bpt / tokens_per_word
            bpb = bpc / bits_per_char_factor
            perplexity = 2 ** entropy
        else:
            bpt = bpc = bpb = perplexity = None

        rows.append({
            "step": log.get("step"),
            "epoch": log.get("epoch"),
            "train_loss": log.get("loss"),
            "train_entropy": log.get("entropy"),
            "train_accuracy": log.get("mean_token_accuracy"),
            "eval_loss": log.get("eval_loss"),
            "eval_entropy": log.get("eval_entropy"),
            "eval_accuracy": log.get("eval_mean_token_accuracy"),
            "perplexity": perplexity,
            "bits_per_token": bpt,
            "bits_per_character": bpc,
            "bits_per_byte": bpb,
        })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv), exist_ok = True)
    df.to_csv(output_csv, index = False)

    return df, last_checkpoint


#----------------------------------------------------------------------------------------------------------------------------#