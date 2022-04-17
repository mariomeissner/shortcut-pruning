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
from torch.nn.functional import log_softmax
from src.datamodules.squad_datamodule import SquadDataModule
from src.models.hf_model import SequenceClassificationTransformer
from src.models.squad_model import QuestionAnsweringTransformer
from src.models.hf_model_pruned import PruningTransformer


def extract_preds(
    checkpoint_path: str,
    save_path: str,
):

    model = QuestionAnsweringTransformer.load_from_checkpoint(checkpoint_path, use_bias_probs=False)
    datamodule = SquadDataModule(
        data_path="/home/meissner/shortcut-pruning/data/squad/",
        tokenizer_name=model.hparams.huggingface_model,
        n_best_size=20,
        max_length=384,
        doc_stride=128,
        max_answer_length=30,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader_noshuffle()

    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    predictions = {"start_logits": {}, "end_logits": {}}
    for idx, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch)
            start_logits = log_softmax(outputs.start_logits, dim=1).detach().cpu().tolist()
            end_logits = log_softmax(outputs.end_logits, dim=1).detach().cpu().tolist()
            idxs = batch["idx"].detach().cpu().tolist()
            for idx, start_logit, end_logit in zip(idxs, start_logits, end_logits):
                predictions["start_logits"][idx] = start_logit
                predictions["end_logits"][idx] = end_logit
            # import ipdb; ipdb.set_trace()

    with open(save_path, "w") as _file:
        _file.write(json.dumps(predictions))

    print(f"Saved predictions to {save_path}.")


if __name__ == "__main__":
    fire.Fire(extract_preds)
