import glob
import json
from multiprocessing.sharedctypes import Value
import os
from pathlib import Path
import fire

import numpy as np
from scipy.special import softmax


def ensemble_outputs(folder: str, method: str = "median"):

    if not method in ("median", "mean"):
        raise ValueError("Unknown method.")

    path = Path(folder)
    outputs = []
    mean_outputs = {}

    for filename in glob.glob(str(path / "*.json")):
        with open(filename, 'r') as _file:
            outputs.append(json.load(_file))

    for key in outputs[0]:
        
        zip_outputs = [output_dict[key] for output_dict in outputs]
        array = np.array(zip_outputs)
        if method == "median":
            array = np.median(array, axis=0)
        elif method == "mean":
            array = np.mean(array, axis=0)
        mean_output = array / array.sum()
        mean_outputs[key] = mean_output.tolist()

    print(json.dumps(mean_outputs))

if __name__ == "__main__":
    fire.Fire(ensemble_outputs)
            