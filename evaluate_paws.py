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
from src.datamodules.fever_datamodule import FeverDataModule
from src.datamodules.qqp_datamodule import QQPDataModule
from src.models.hf_model import SequenceClassificationTransformer
from src.models.hf_model_pruned import PruningTransformer


def evaluate_on_fever(
    checkpoint_path: str = None,
    paste: str = None,
    pruned_model: bool = False,
    max_length: int = 128,
    batch_size: int = 256,
):

    if checkpoint_path:
        path_list = [checkpoint_path]
    elif paste:
        path_list = [line.strip() + "/checkpoints/last.ckpt" for line in paste.split("\n")]

    if pruned_model:
        print("Loading PruningTransformer instead of the normal one.")
        model_class = PruningTransformer
    else:
        model_class = SequenceClassificationTransformer

    model = model_class.load_from_checkpoint(path_list[0], use_bias_probs=False)

    # Load dataset
    qqp_datamodule = QQPDataModule(
        2,
        "question1",
        "question2",
        "/home/meissner/shortcut-pruning/data/paws/paws_qqp/",
        tokenizer_name=model.hparams.huggingface_model,
    )
    qqp_datamodule.setup()
    dataloader = qqp_datamodule.test_dataloader()[0]

    result_string = []

    for path in path_list:

        model = model_class.load_from_checkpoint(path, use_bias_probs=False)
        model.eval()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # Run evaluation
        predictions = []
        targets = []
        for idx, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits, preds = model(batch)
                preds = list(preds.detach().cpu())
            predictions.extend(preds)
            targets.extend(batch["labels"])

        predictions = np.array(predictions)
        targets = np.array(targets)
        bools = predictions == targets
        total_acc = bools.mean()
        zero_acc = bools[targets == 0].mean()
        one_acc = bools[targets == 1].mean()

        result_string.append(f"{total_acc}, {zero_acc}, {one_acc}")

    print("\n".join(result_string))


if __name__ == "__main__":
    fire.Fire(evaluate_on_fever)
