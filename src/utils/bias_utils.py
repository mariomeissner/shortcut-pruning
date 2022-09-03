import json
import pickle

# temporary hack for the pythonroot issue
from os.path import exists
from typing import Dict

import numpy as np

# import src.utils.bias_utils as bias_utils


# def load_bias(bias_name, custom_path=None) -> Dict[str, np.ndarray]:
#     """Load dictionary of example_id->bias where bias is a length 3 array
#     of log-probabilities"""

#     if custom_path is not None:  # file contains probs
#         with open(custom_path, "r") as bias_file:
#             all_lines = bias_file.read()
#             bias = json.loads(all_lines)
#             for k, v in bias.items():
#                 bias[k] = np.log(np.array(v))
#         return bias

#     if bias_name == "hans":
#         if bias_name == "hans":
#             bias_src = paths.MNLI_WORD_OVERLAP_BIAS
#         if not exists(bias_src):
#             raise Exception("lexical overlap bias file is not found")
#         bias = bias_utils.load_pickle(bias_src)
#         for k, v in bias.items():
#             # Convert from entail vs non-entail to 3-way classes by splitting non-entail
#             # to neutral and contradict
#             bias[k] = np.array(
#                 [
#                     v[0] - np.log(2.0),
#                     v[1],
#                     v[0] - np.log(2.0),
#                 ]
#             )
#         return bias

#     if bias_name in paths.BIAS_SOURCES:
#         file_path = paths.BIAS_SOURCES[bias_name]
#         with open(file_path, "r") as hypo_file:
#             all_lines = hypo_file.read()
#             bias = json.loads(all_lines)
#             for k, v in bias.items():
#                 bias[k] = np.array(v)
#         return bias
#     else:
#         raise Exception("invalid bias name")


def load_bias_probs(file_path: str):

    with open(file_path, "r") as teacher_file:
        all_lines = teacher_file.read()
        all_json = json.loads(all_lines)

    return all_json
