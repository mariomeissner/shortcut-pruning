import argparse
from pathlib import Path
from torch._C import Argument
import transformers
from transformers import BertTokenizer, AutoTokenizer
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from argparse import ArgumentParser


def main():

    transformers.logging.set_verbosity_error()

    parser = ArgumentParser()
    parser.add_argument("which", type=str, choices=['snli', 'mnli', 'hans'])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_procs", type=int, default=8)
    args = parser.parse_args()

    if args.which == 'snli':
        dataset : DatasetDict = load_dataset("snli")
    elif args.which == 'mnli':
        dataset : DatasetDict = load_dataset("glue", "mnli")
    else:
        dataset : DatasetDict = load_dataset("hans")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess_function(examples: dict):
        sents = (examples["premise"], examples["hypothesis"])
        result = tokenizer(*sents, max_length=args.max_length, truncation="longest_first")
        return result

    dataset = dataset.map(preprocess_function, batched=True, num_proc=args.num_procs)
    dataset = dataset.filter(lambda sample: sample["label"] != -1)
    dataset = dataset.rename_column("label", "labels")
    # dataset = dataset.remove_columns(["premise", "hypothesis"])
    # dataset.set_format("torch")
    dataset.save_to_disk(args.output_dir)

    print(f"Saved processed {args.which} dataset into {args.output_dir} folder.")

if __name__ == "__main__":
    main()