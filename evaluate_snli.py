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


def evaluate_on_snli(
    checkpoint_path: str = None,
    checkpoint_list_file: str = None,
    pruned_model: bool = False,
    do_negation_subsets: bool = False,
    do_lex_overlap_subsets: bool = False,
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
        snli: DatasetDict = datasets.load_dataset("snli")
        # Remove unnecessary splits for speed
        snli = DatasetDict({"train": snli["train"], "validation": snli["validation"]})
        tokenizer = AutoTokenizer.from_pretrained(model.hparams.huggingface_model)

        def preprocess_func(examples: dict):
            args = (examples["premise"], examples["hypothesis"])
            result = tokenizer(*args, max_length=max_length, truncation=True)
            return result

        snli = snli.map(preprocess_func, batched=True, num_proc=4)
        snli.set_format(
            "torch",
            columns=[
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "label",
                # "idx",
            ],
        )

        train_dataloader = DataLoader(
            snli["train"], batch_size=batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
        )
        val_dataloader = DataLoader(
            snli["validation"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        with open("data/subsets/snli_negation_indices.json") as _file:
            negation_indices = json.load(_file)
        with open("data/subsets/snli_lex_overlap_indices.json") as _file:
            overlap_indices = json.load(_file)

        def evaluate_subset(predictions, targets, cont_indices, ent_indices):
            cont_acc = np.mean(predictions[cont_indices] == targets[cont_indices])
            ent_acc = np.mean(predictions[ent_indices] == targets[ent_indices])
            return cont_acc, ent_acc

        results = []
        predictions = []
        output_logits = []
        for idx, batch in enumerate(tqdm(val_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits, preds = model(batch)
                preds = preds.detach().cpu().tolist()
                logits = logits.detach().cpu().tolist()
            predictions.extend(preds)
            output_logits.extend(logits)

        predictions = np.array(predictions)
        output_logits = np.array(output_logits)
        targets = np.array(snli["validation"]["label"])
        accuracy = np.mean(predictions == targets)
        results.append(accuracy)

        if do_negation_subsets:
            neg_cont_acc, neg_ent_acc = evaluate_subset(
                predictions, targets, negation_indices["val_conts"], negation_indices["val_ents"]
            )
            neg_cont_cont_level = output_logits[negation_indices["val_conts"], 2].mean()
            neg_cont_ent_level = output_logits[negation_indices["val_conts"], 0].mean()
            neg_ent_cont_level = output_logits[negation_indices["val_ents"], 2].mean()
            neg_ent_ent_level = output_logits[negation_indices["val_ents"], 0].mean()
            results.extend([neg_cont_acc, neg_ent_acc, neg_cont_cont_level, neg_cont_ent_level, neg_ent_cont_level, neg_ent_cont_level, neg_ent_ent_level])

        if do_lex_overlap_subsets:
            lex_cont_acc, lex_ent_acc = evaluate_subset(
                predictions, targets, overlap_indices["val_conts"], overlap_indices["val_ents"]
            )
            lex_cont_cont_level = output_logits[overlap_indices["val_conts"], 2].mean()
            lex_cont_ent_level = output_logits[overlap_indices["val_conts"], 0].mean()
            lex_ent_cont_level = output_logits[overlap_indices["val_ents"], 2].mean()
            lex_ent_ent_level = output_logits[overlap_indices["val_ents"], 0].mean()
            results.extend([lex_cont_acc, lex_ent_acc, lex_cont_cont_level, lex_cont_ent_level, lex_ent_cont_level, lex_ent_cont_level, lex_ent_ent_level])


        print(",".join([str(result) for result in results]))


if __name__ == "__main__":
    fire.Fire(evaluate_on_snli)
