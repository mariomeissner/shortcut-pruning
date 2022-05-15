import importlib
from multiprocessing.sharedctypes import Value
from typing import Sequence, Tuple
from datasets.dataset_dict import DatasetDict, Dataset
import fire
import datasets
import json

import numpy as np
from transformers import AutoTokenizer
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import torch
from torch.nn.functional import softmax
from src.models.hf_model import SequenceClassificationTransformer
from src.models.hf_model_pruned import PruningTransformer


FEVER_LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}


def extract_preds(
    checkpoint_path: str,
    save_path: str,
    dataset_name: str,
    pruned_model: bool = False,
    subset: str = "train",
    max_length: int = 128,
    batch_size: int = 256,
):

    if dataset_name == "mnli":
        dataset = datasets.load_dataset("glue", "mnli").rename_column("label", "labels")
        sent1 = "premise"
        sent2 = "hypothesis"
    elif dataset_name == "qqp":
        dataset = datasets.load_dataset("glue", "qqp").rename_column("label", "labels")
        sent1 = "question1"
        sent2 = "question2"
    elif dataset_name == "fever":
        dataset = datasets.load_dataset("json", data_files="data/fever/fever_train.jsonl")
        dataset = (
            dataset.map(lambda sample: {"labels": FEVER_LABEL_MAP[sample["gold_label"]]})
            .remove_columns(["weight"])
            .rename_column("id", "idx")
        )
        sent1 = "evidence"
        sent2 = "claim"
    else:
        raise ValueError("Unrecognized dataset name.")

    if pruned_model:
        print("Loading PruningTransformer instead of the normal one.")
        model_class = PruningTransformer
    else:
        model_class = SequenceClassificationTransformer

    model = model_class.load_from_checkpoint(checkpoint_path, use_bias_probs=False)
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.huggingface_model)

    def preprocess_func(examples: dict):
        args = (examples[sent1], examples[sent2])
        result = tokenizer(*args, max_length=max_length, truncation=True)
        return result

    dataset = dataset.map(preprocess_func, batched=True, num_proc=4)

    dataset.set_format(
        "torch",
        columns=[
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "idx",
        ],
    )

    dataloader = DataLoader(
        dataset[subset],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    predictions = {}
    for idx, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits, preds = model(batch)
            soft_preds = softmax(logits, dim=1).detach().cpu().tolist()
            idxs = batch["idx"].detach().cpu().tolist()
            for idx, soft_pred in zip(idxs, soft_preds):
                predictions[idx] = soft_pred

    with open(save_path, "w") as _file:
        _file.write(json.dumps(predictions))

    print(f"Saved predictions to {save_path}.")


if __name__ == "__main__":
    fire.Fire(extract_preds)
