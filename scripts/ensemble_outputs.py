import glob
import json
import os
from pathlib import Path
import sys
import fire
from tqdm import tqdm

import numpy as np
from scipy.special import softmax


def ensemble_outputs(folder: str, method: str = "median", squad=False):

    if not method in ("median", "mean", "max"):
        raise ValueError("Unknown method.")

    path = Path(folder)
    outputs = []

    for filename in glob.glob(str(path / "*.json")):
        with open(filename, "r") as _file:
            print(f"Loading file {filename}", file=sys.stderr)
            outputs.append(json.load(_file))

    def get_mean_outputs(outputs):
        mean_outputs = {}
        for key in tqdm(outputs[0]):
            zip_outputs = [output_dict[key] for output_dict in outputs]
            array = np.array(zip_outputs)
            if method == "median":
                array = np.median(array, axis=0)
            elif method == "mean":
                array = np.mean(array, axis=0)
            elif method == "max":
                array = np.max(array, axis=0)
            # array = array / array.sum()
            mean_outputs[key] = array.tolist()
        return mean_outputs

    if squad:
        start_logit_outputs = [output["start_logits"] for output in outputs]
        end_logit_outputs = [output["end_logits"] for output in outputs]
        start_mean_outputs = get_mean_outputs(start_logit_outputs)
        end_mean_outputs = get_mean_outputs(end_logit_outputs)
        mean_outputs = {"start_logits": start_mean_outputs, "end_logits": end_mean_outputs}
    else:
        mean_outputs = get_mean_outputs(outputs)

    print(json.dumps(mean_outputs))


if __name__ == "__main__":
    fire.Fire(ensemble_outputs)
