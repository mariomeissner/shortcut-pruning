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

def evaluate_on_mnli(
    checkpoint_path: str,
    pruned_model: bool = False,
    data_path: str = "data/mnli-tokenized",
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
    mnli: DatasetDict = datasets.load_dataset("glue", "mnli")
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.huggingface_model)

    def preprocess_func(examples: dict):
        args = (examples["premise"], examples["hypothesis"])
        result = tokenizer(*args, max_length=max_length, truncation=True)
        return result

    mnli = mnli.map(preprocess_func, batched=True, num_proc=4)
    mnli = mnli.rename_column("label", "labels")
    mnli.set_format(
        "torch",
        columns=[
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
        ],
    )
    matched_dataloader = DataLoader(
        mnli["validation_matched"], batch_size=batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    mismatched_dataloader = DataLoader(
        mnli["validation_mismatched"], batch_size=batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Run matched evaluation
    predictions = []
    for idx, batch in enumerate(tqdm(matched_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits, preds = model(batch)
            preds = list(preds.detach().cpu())
        predictions.extend(preds)

    predictions = np.array(predictions)
    targets = np.array(mnli["validation_matched"]["labels"])
    accuracy = np.mean(predictions == targets)
    print(f"MNLI Validation Matched score: {accuracy}")

    # Run mismatched evaluation
    predictions = []
    for idx, batch in enumerate(tqdm(mismatched_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits, preds = model(batch)
            preds = list(preds.detach().cpu())
        predictions.extend(preds)

    predictions = np.array(predictions)
    targets = np.array(mnli["validation_mismatched"]["labels"])
    accuracy = np.mean(predictions == targets)
    print(f"MNLI Validation Mismatched score: {accuracy}")


if __name__ == "__main__":
    fire.Fire(evaluate_on_mnli)
