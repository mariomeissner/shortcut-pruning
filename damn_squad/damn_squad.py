import json


import json

from utils_squad import read_squad_examples

path = "../data/squad/add_sent.json"
# with open("../data/squad/add_sent.json") as _file:
#     addsent = json.load(_file)

add_sent = read_squad_examples(path, is_training=False, version_2_with_negative=False)
add_sent_dict = {
    "id" : [sample.qas_id for sample in add_sent],
    "doc_tokens" : [sample.doc_tokens for sample in add_sent],
    "context" : [" ".]
}