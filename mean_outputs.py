import glob
import json
import os
from pathlib import Path
import fire

import numpy as np
from scipy.special import softmax


def mean_outputs_func(folder: str):
    path = Path(folder)
    outputs = []
    mean_outputs = {}

    for filename in glob.glob(str(path / "*.json")):
        with open(filename, 'r') as _file:
            outputs.append(json.load(_file))

    for key in outputs[0]:
        
        zip_outputs = [output_dict[key] for output_dict in outputs]
        array = np.array(zip_outputs)
        array = np.median(array, axis=0)
        mean_output = array / array.sum()
        mean_outputs[key] = mean_output.tolist()

    print(json.dumps(mean_outputs))

if __name__ == "__main__":
    fire.Fire(mean_outputs_func)
            