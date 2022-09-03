import json
import fire

def convert(filename: str):

    with open(filename, "r") as _file:
        preds = json.load(_file)

    new_preds = {}

    for key, value in preds.items():
        # UKPLab repo preds are contradiction, entailment, neutral
        # We want entailment, neutral, contradiction
        new_preds[key] = [value[1], value[2], value[0]]

    with open(filename + ".fixed", "w") as _file:
        json.dump(new_preds, _file)

    print("Done!")

if __name__ == "__main__":
    fire.Fire(convert)

