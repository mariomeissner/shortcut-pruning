from pathlib import Path
from typing import List

import subprocess
import fire

TAIL_PATH = "/checkpoints/last.ckpt"


def run(
    checkpoint_basepath: str,
    prediction_savepath: str,
    prediction_savename: str,
    dataset: str,
    seeds: List[int] = [10, 11, 12, 13, 14],
):

    for i, seed in enumerate(seeds):
        print(f"Working on seed {seed}.")
        checkpoint = Path(checkpoint_basepath) / str(i) / TAIL_PATH
        save_path = str(Path(prediction_savepath) / prediction_savename) + "-" + str(seed) + ".json"
        try:
            completed = subprocess.run(["python", "extract_preds.py", checkpoint, save_path, "--dataset_name", dataset])
        except subprocess.CalledProcessError as error:
            print(error.output)
            raise error
        print(f"Successfully extracted preds into {save_path}")


if __name__ == "__main__":
    fire.Fire(run)
