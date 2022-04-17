from asyncio import subprocess
from pathlib import Path
from typing import List

import fire

TAIL_PATH = "/checkpoints/last.ckpt"


def run(
    checkpoint_basepath: str,
    prediction_savepath: str,
    prediction_savename: str,
    dataset: str,
    seeds: List[int] = [10, 11, 12, 13],
):

    for i, seed in enumerate(seeds):
        print(f"Working on seed {seed}.")
        checkpoint = Path(checkpoint_basepath) / str(i) / TAIL_PATH
        save_path = Path(prediction_savepath) / prediction_savename + "-", str(seed) + ".json"
        completed = subprocess.run("python", "extract_preds.py", checkpoint, save_path, "--dataset", dataset)
        completed.check_returncode()
        print(f"Successfully extracted preds into {save_path}")


if __name__ == "__main__":
    fire.Fire(run)
