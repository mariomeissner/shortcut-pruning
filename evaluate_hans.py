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
from .src.models.hf_model import SequenceClassificationTransformer

def evaluate_on_hans(
    module_path: str,
    class_name: str,
    checkpoint_path: str,
    data_path: str = "data/hans-tokenized",
    max_length: int = 128,
):

    # Load model
    # TODO: Fix this, it breaks due to relative import failure (from scripts folder)
    # module = importlib.import_module(module_path)
    # model_class : LightningModule = getattr(module, class_name)

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
        hans["validation"], batch_size=64, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Run evaluation
    predictions = []
    for idx, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(batch)
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
        predictions.extend(preds)

    predictions = np.array(predictions)
    targets = np.array(hans["validation"]["labels"])

    accuracy = np.mean(predictions == targets)
    print(f"Hans validation score: {accuracy}.")


if __name__ == "__main__":
    fire.Fire(evaluate_on_hans)
