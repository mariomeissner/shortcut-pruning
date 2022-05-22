from glob import glob
import os
import importlib
import sys
from typing import Sequence
from datasets.dataset_dict import DatasetDict
import fire
import datasets
import json

import numpy as np
from sklearn.utils import shuffle
from transformers import AutoTokenizer
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import torch
from torch.nn.functional import softmax
from src.datamodules.snli_datamodule import SNLIDataModule
from src.datamodules.mnli_datamodule import MNLIDataModule
from src.models.hf_model import SequenceClassificationTransformer


def total_evaluation(
    checkpoint_path: str = None,
    multirun_folder: str = None,
    checkpoint_list_file: str = None,
    recorder_checkpoint_path: str = None,
):
    def get_immediate_subdirectories(a_dir):
        return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

    # Find model paths
    if checkpoint_path:
        path_list = [checkpoint_path]
    elif multirun_folder:
        runs = get_immediate_subdirectories(multirun_folder)
        runs = list(sorted([int(run) for run in runs]))
        path_list = [multirun_folder + f"{int(run)}/checkpoints/last.ckpt" for run in runs]
    elif checkpoint_list_file:
        with open(checkpoint_list_file) as _file:
            path_list = _file.readlines()
            path_list = [line.strip() + "/checkpoints/last.ckpt" for line in path_list]
    elif recorder_checkpoint_path:
        checkpoints = glob(recorder_checkpoint_path + "*.ckpt")
        checkpoints = list(sorted([int(name.split("-")[-1].split(".")[0]) for name in checkpoints]))
        checkpoints = [f"0-{number}.ckpt" for number in checkpoints]
        path_list = [recorder_checkpoint_path + checkpoint for checkpoint in checkpoints]
    else:
        raise ValueError("Provide either of checkpoint_path or checkpoint_list_file!")

    # Load first model to get tokenizer
    print(f"Getting tokenizer name from first model.", file=sys.stderr)
    model = SequenceClassificationTransformer.load_from_checkpoint(path_list[0], use_bias_probs=False)

    # Get dataloaders
    snli_datamodule = SNLIDataModule(
        3, sentence_1_name="premise", sentence_2_name="hypothesis", tokenizer_name=model.hparams.huggingface_model
    )
    mnli_datamodule = MNLIDataModule(
        3,
        sentence_1_name="premise",
        sentence_2_name="hypothesis",
        data_path="data/mnli/",
        tokenizer_name=model.hparams.huggingface_model,
    )
    snli_dataset = datasets.load_dataset("snli")
    mnli_dataset = datasets.load_dataset("glue", "mnli")
    hans_dataset = datasets.load_dataset("hans")
    snli_datamodule.setup()
    mnli_datamodule.setup()
    snli_val_loader = snli_datamodule.val_dataloader()
    mnli_val_m_loader = mnli_datamodule.val_dataloader()
    mnli_val_mm_loader = mnli_datamodule.test_dataloader()[0]
    hans_loader = mnli_datamodule.test_dataloader()[1]

    # Load subset indices
    with open("data/subsets/mnli_negation_indices.json") as _file:
        mnli_negation_indices = json.load(_file)
    with open("data/subsets/mnli_lex_overlap_indices.json") as _file:
        mnli_overlap_indices = json.load(_file)
    with open("data/subsets/snli_negation_indices.json") as _file:
        snli_negation_indices = json.load(_file)
    with open("data/subsets/snli_lex_overlap_indices.json") as _file:
        snli_overlap_indices = json.load(_file)
    with open("data/subsets/mnli_z_statistic_indices.json") as _file:
        mnli_z_statistic_indices = json.load(_file)

    def get_predictions(model: torch.nn.Module, loader: DataLoader):
        logits = []
        predictions = []
        for idx, batch in enumerate(tqdm(loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                batch_logits, batch_preds = model(batch)
                batch_logits = batch_logits.detach().cpu().tolist()
                batch_preds = batch_preds.detach().cpu().tolist()
            logits.extend(batch_logits)
            predictions.extend(batch_preds)
        return np.array(logits), np.array(predictions)

    # Subset evaluation function
    def evaluate_subset(predictions, targets, cont_indices, ent_indices):
        cont_acc = np.mean(predictions[cont_indices] == targets[cont_indices])
        ent_acc = np.mean(predictions[ent_indices] == targets[ent_indices])
        return cont_acc, ent_acc

    print(
        ",".join(
            [
                # "MNLI ValM Acc",
                # "MNLI ValM NegContLevel",
                # "MNLI ValM NegEntLevel",
                # "MNLI ValM NegContAcc",
                # "MNLI ValM NegEntAcc",
                # "MNLI ValM LexContLevel",
                # "MNLI ValM LexEntLevel",
                # "MNLI ValM LexContAcc",
                # "MNLI ValM LexEntAcc",
                # "MNLI ValMM Acc",
                # "MNLI ValMM NegContLevel",
                # "MNLI ValMM NegEntLevel",
                # "MNLI ValMM NegContAcc",
                # "MNLI ValMM NegEntAcc",
                # "MNLI ValMM LexContLevel",
                # "MNLI ValMM LexEntLevel",
                # "MNLI ValMM LexContAcc",
                # "MNLI ValMM LexEntAcc",
                # "SNLI Val Acc",
                # "SNLI Val NegContLevel",
                # "SNLI Val NegEntLevel",
                # "SNLI Val NegContAcc",
                # "SNLI Val NegEntAcc",
                # "SNLI Val LexContLevel",
                # "SNLI Val LexEntLevel",
                # "SNLI Val LexContAcc",
                # "SNLI Val LexEntAcc",
                # "HANS Val Acc",
                # "HANS Val LexContLevel",
                # "HANS Val LexEntLevel",
                # "HANS Val LexContAcc",
                # "HANS Val LexEntAcc",
                # "HANS Val SubContLevel",
                # "HANS Val SubEntLevel",
                # "HANS Val SubContAcc",
                # "HANS Val SubEntAcc",
                # "HANS Val ConstContLevel",
                # "HANS Val ConstLexEntLevel",
                # "HANS Val ConstLexContAcc",
                # "HANS Val ConstLexEntAcc",
                "MNLI Val ZContBiasAcc",
                "MNLI Val ZContAntiBiasAcc",
                "MNLI Val ZEntBiasAcc",
                "MNLI Val ZEntAntiBiasAcc",
            ]
        )
    )
    # Evaluation loop
    for path in path_list:

        print(f"Loading checkpoint from: {path}", file=sys.stderr)
        model = SequenceClassificationTransformer.load_from_checkpoint(path, use_bias_probs=False)
        model.eval()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        results = []

        # # Run MNLI matched evaluation
        # logits, predictions = get_predictions(model, mnli_val_m_loader)
        # targets = np.array(mnli_dataset["validation_matched"]["label"])
        # accuracy = np.mean(predictions == targets)
        # results.append(accuracy)
        # # Negation subset (bias towards cont)
        # neg_cont_acc, neg_ent_acc = evaluate_subset(
        #     predictions, targets, mnli_negation_indices["val_m_conts"], mnli_negation_indices["val_m_ents"]
        # )
        # neg_cont_level = np.mean(logits[mnli_negation_indices["val_m_conts"], 2])
        # neg_ent_level = np.mean(logits[mnli_negation_indices["val_m_ents"], 2])
        # results.extend([neg_cont_level, neg_ent_level])
        # results.extend([neg_cont_acc, neg_ent_acc])
        # # Lexical Overlap subset (bias towards entail)
        # lex_cont_acc, lex_ent_acc = evaluate_subset(
        #     predictions, targets, mnli_overlap_indices["val_m_conts"], mnli_overlap_indices["val_m_ents"]
        # )
        # lex_cont_level = np.mean(logits[mnli_overlap_indices["val_m_conts"], 0])
        # lex_ent_level = np.mean(logits[mnli_overlap_indices["val_m_ents"], 0])
        # results.extend([lex_cont_level, lex_ent_level])
        # results.extend([lex_cont_acc, lex_ent_acc])

        # # Run MNLI mismatched evaluation
        # logits, predictions = get_predictions(model, mnli_val_mm_loader)
        # targets = np.array(mnli_dataset["validation_mismatched"]["label"])
        # accuracy = np.mean(predictions == targets)
        # results.append(accuracy)
        # # Negation subset (bias towards cont)
        # neg_cont_acc, neg_ent_acc = evaluate_subset(
        #     predictions, targets, mnli_negation_indices["val_mm_conts"], mnli_negation_indices["val_mm_ents"]
        # )
        # neg_cont_level = np.mean(logits[mnli_negation_indices["val_mm_conts"], 2])
        # neg_ent_level = np.mean(logits[mnli_negation_indices["val_mm_ents"], 2])
        # results.extend([neg_cont_level, neg_ent_level])
        # results.extend([neg_cont_acc, neg_ent_acc])
        # # Lexical Overlap subset (bias towards entail)
        # lex_cont_acc, lex_ent_acc = evaluate_subset(
        #     predictions, targets, mnli_overlap_indices["val_mm_conts"], mnli_overlap_indices["val_mm_ents"]
        # )
        # lex_cont_level = np.mean(logits[mnli_overlap_indices["val_mm_conts"], 0])
        # lex_ent_level = np.mean(logits[mnli_overlap_indices["val_mm_ents"], 0])
        # results.extend([lex_cont_level, lex_ent_level])
        # results.extend([lex_cont_acc, lex_ent_acc])

        # # Run SNLI evaluation
        # logits, predictions = get_predictions(model, snli_val_loader)
        # targets = np.array(snli_dataset["validation"]["label"])
        # accuracy = np.mean(predictions == targets)
        # results.append(accuracy)
        # # Negation subset (bias towards cont)
        # neg_cont_acc, neg_ent_acc = evaluate_subset(
        #     predictions, targets, snli_negation_indices["val_conts"], snli_negation_indices["val_ents"]
        # )
        # neg_cont_level = np.mean(logits[snli_negation_indices["val_conts"], 2])
        # neg_ent_level = np.mean(logits[snli_negation_indices["val_ents"], 2])
        # results.extend([neg_cont_level, neg_ent_level])
        # results.extend([neg_cont_acc, neg_ent_acc])
        # # Lexical Overlap subset (bias towards entail)
        # lex_cont_acc, lex_ent_acc = evaluate_subset(
        #     predictions, targets, snli_overlap_indices["val_conts"], snli_overlap_indices["val_ents"]
        # )
        # lex_cont_level = np.mean(logits[snli_overlap_indices["val_conts"], 0])
        # lex_ent_level = np.mean(logits[snli_overlap_indices["val_ents"], 0])
        # results.extend([lex_cont_level, lex_ent_level])
        # results.extend([lex_cont_acc, lex_ent_acc])

        # # Run Hans evaluation
        # logits, predictions = get_predictions(model, hans_loader)
        # predictions[predictions == 2] = 1
        # logits = np.concatenate(
        #     [logits[:, 0:1], logits[:, 1:].max(axis=1, keepdims=True)], axis=1
        # )  # max between neutral and cont
        # targets = np.array(hans_dataset["validation"]["label"])
        # heuristic = np.array(hans_dataset["validation"]["heuristic"])
        # accuracy_bools = predictions == targets
        # accuracy = np.mean(accuracy_bools)
        # results.append(accuracy)

        # # lexical overlap
        # lex_cont_indices = (heuristic == "lexical_overlap") & (targets == 1)
        # lex_ent_indices = (heuristic == "lexical_overlap") & (targets == 0)
        # lex_cont_level = np.mean(logits[lex_cont_indices, 0])
        # lex_ent_level = np.mean(logits[lex_ent_indices, 0])
        # lex_cont_acc = np.mean(accuracy_bools[lex_cont_indices])
        # lex_ent_acc = np.mean(accuracy_bools[lex_ent_indices])
        # results.extend([lex_cont_level, lex_ent_level, lex_cont_acc, lex_ent_acc])

        # # subsequence
        # sub_cont_indices = (heuristic == "subsequence") & (targets == 1)
        # sub_ent_indices = (heuristic == "subsequence") & (targets == 0)
        # sub_cont_level = np.mean(logits[sub_cont_indices, 0])
        # sub_ent_level = np.mean(logits[sub_ent_indices, 0])
        # sub_cont_acc = np.mean(accuracy_bools[sub_cont_indices])
        # sub_ent_acc = np.mean(accuracy_bools[sub_ent_indices])
        # results.extend([sub_cont_level, sub_ent_level, sub_cont_acc, sub_ent_acc])

        # # constituent
        # constitu_cont_indices = (heuristic == "constituent") & (targets == 1)
        # constitu_ent_indices = (heuristic == "constituent") & (targets == 0)
        # constitu_cont_level = np.mean(logits[constitu_cont_indices, 0])
        # constitu_ent_level = np.mean(logits[constitu_ent_indices, 0])
        # constitu_cont_acc = np.mean(accuracy_bools[constitu_cont_indices])
        # constitu_ent_acc = np.mean(accuracy_bools[constitu_ent_indices])
        # results.extend([constitu_cont_level, constitu_ent_level, constitu_cont_acc, constitu_ent_acc])

        # Z Statistic
        # Run MNLI matched evaluation
        logits, predictions = get_predictions(model, mnli_val_m_loader)
        targets = np.array(mnli_dataset["validation_matched"]["label"])
        acc_bools = predictions == targets
        # Contradiction subset
        cont_bias_acc = acc_bools[mnli_z_statistic_indices["cont_bias_indices"]].mean()
        cont_antibias_acc = acc_bools[mnli_z_statistic_indices["cont_antibias_indices"]].mean()
        ent_bias_acc = acc_bools[mnli_z_statistic_indices["ent_bias_indices"]].mean()
        ent_antibias_acc = acc_bools[mnli_z_statistic_indices["ent_antibias_indices"]].mean()
        results.extend([cont_bias_acc, cont_antibias_acc, ent_bias_acc, ent_antibias_acc])

        print(",".join([str(result) for result in results]))


if __name__ == "__main__":
    fire.Fire(total_evaluation)
