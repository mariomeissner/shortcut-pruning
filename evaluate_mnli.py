import importlib
import sys
from typing import Sequence
from datasets.dataset_dict import DatasetDict
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


def evaluate_on_mnli(
    checkpoint_path: str = None,
    checkpoint_list_file: str = None,
    pruned_model: bool = False,
    do_matched: bool = True,
    do_mismatched: bool = False,
    do_negation_subsets: bool = False,
    do_lex_overlap_subsets: bool = False,
    do_new_neg_ent_subset: bool = False,
    max_length: int = 128,
    batch_size: int = 256,
):

    if checkpoint_path:
        path_list = [checkpoint_path]
    elif checkpoint_list_file:
        with open(checkpoint_list_file) as _file:
            path_list = _file.readlines()
            path_list = [line.strip() + "/checkpoints/last.ckpt" for line in path_list]
    else:
        raise ValueError("Provide either of checkpoint_path or checkpoint_list_file!")

    for path in path_list:

        if pruned_model:
            print("Loading PruningTransformer instead of the normal one.", file=sys.stderr)
            model_class = PruningTransformer
        else:
            model_class = SequenceClassificationTransformer

        print(f"Loading model from {path}.", file=sys.stderr)
        model = model_class.load_from_checkpoint(path, use_bias_probs=False)
        model.eval()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # Load dataset
        mnli: DatasetDict = datasets.load_dataset("glue", "mnli")
        # Remove unnecessary splits for speed
        mnli = DatasetDict(
            {"validation_matched": mnli["validation_matched"], "validation_mismatched": mnli["validation_mismatched"]}
        )
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
                "idx",
            ],
        )
        # train_dataloader = DataLoader(
        #     mnli["train"], batch_size=batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
        # )
        matched_dataloader = DataLoader(
            mnli["validation_matched"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        )
        mismatched_dataloader = DataLoader(
            mnli["validation_mismatched"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        with open("data/subsets/mnli_negation_indices.json") as _file:
            negation_indices = json.load(_file)
        with open("data/subsets/mnli_lex_overlap_indices.json") as _file:
            overlap_indices = json.load(_file)
        with open("data/subsets/mnli_neg_ent_new_indices.json") as _file:
            new_neg_ent_indices = json.load(_file)

        def evaluate_subset(predictions, targets, cont_indices, ent_indices):
            all_indices = cont_indices + ent_indices
            cont_acc = np.mean(predictions[cont_indices] == targets[cont_indices])
            ent_acc = np.mean(predictions[ent_indices] == targets[ent_indices])
            all_acc = np.mean(predictions[all_indices] == targets[all_indices])
            return cont_acc, ent_acc, all_acc

        results = []
        # Run matched evaluation
        if do_matched:
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
            results.append(accuracy)

            if do_negation_subsets:
                neg_cont_acc, neg_ent_acc, neg_acc = evaluate_subset(
                    predictions, targets, negation_indices["val_m_conts"], negation_indices["val_m_ents"]
                )
                results.extend([neg_cont_acc, neg_ent_acc, neg_acc])

            if do_lex_overlap_subsets:
                lex_cont_acc, lex_ent_acc, lex_acc = evaluate_subset(
                    predictions, targets, overlap_indices["val_m_conts"], overlap_indices["val_m_ents"]
                )
                results.extend([lex_cont_acc, lex_ent_acc, lex_acc])

            if do_new_neg_ent_subset:
                bools = predictions == targets
                results.append(bools[new_neg_ent_indices].mean())

        # Run mismatched evaluation
        if do_mismatched:
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
            results.append(accuracy)

            if do_negation_subsets:
                neg_cont_acc, neg_ent_acc, neg_acc = evaluate_subset(
                    predictions, targets, negation_indices["val_mm_conts"], negation_indices["val_mm_ents"]
                )
                results.extend([neg_cont_acc, neg_ent_acc, neg_acc])

            if do_lex_overlap_subsets:
                lex_cont_acc, lex_ent_acc, lex_acc = evaluate_subset(
                    predictions, targets, overlap_indices["val_mm_conts"], overlap_indices["val_mm_ents"]
                )
                results.extend([lex_cont_acc, lex_ent_acc, lex_acc])

        print(",".join([str(result) for result in results]))


if __name__ == "__main__":
    fire.Fire(evaluate_on_mnli)
