import importlib
from typing import Sequence
from datasets.dataset_dict import DatasetDict
import fire
import datasets

import numpy as np
from transformers import AutoTokenizer
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import torch 
from src.models.hf_model import SequenceClassificationTransformer
from src.models.hf_model_pruned import PruningTransformer

def evaluate_on_hans(
    checkpoint_path: str,
    pruned_model: bool = False,
    max_length: int = 128,
    batch_size: int = 256,
):

    if pruned_model:
        print("Loading PruningTransformer instead of the normal one.")
        model_class = PruningTransformer
    else:
        model_class = SequenceClassificationTransformer
    
    model = model_class.load_from_checkpoint(checkpoint_path)

    # Load dataset
    hans: DatasetDict = datasets.load_dataset("hans")
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.huggingface_model)

    def preprocess_func(examples: dict):
        args = (examples["premise"], examples["hypothesis"])
        result = tokenizer(*args, max_length=max_length, truncation=True)
        return result

    hans = hans.map(preprocess_func, batched=True, num_proc=4)
    hans = hans.rename_column("label", "labels")
    hans.set_format(
        "torch",
        columns=[
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
        ],
    )
    dataloader = DataLoader(
        hans["validation"], batch_size=batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Run evaluation
    predictions = []
    for idx, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits, preds = model(batch)
            preds = list(preds.detach().cpu())
        predictions.extend(preds)

    predictions = np.array(predictions)
    # In hans, neutral and contradict go together into non-entail
    predictions[predictions==2] = 1 
    targets = np.array(hans["validation"]["labels"])
    entail_subset = targets == 0
    nonentail_subset = targets == 1
    assert np.sum(entail_subset) + np.sum(nonentail_subset) == len(targets)

    total_accuracy = np.mean(predictions == targets)
    entail_accuracy = np.mean(predictions[entail_subset] == targets[entail_subset])
    nonentail_accuracy = np.mean(predictions[nonentail_subset] == targets[nonentail_subset])
    print(f"Hans total score: {total_accuracy}.")
    print(f"Hans entail score: {entail_accuracy}.")
    print(f"Hans nonentail score: {nonentail_accuracy}.")


if __name__ == "__main__":
    fire.Fire(evaluate_on_hans)
