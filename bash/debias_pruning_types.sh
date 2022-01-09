#!/bin/bash
set -o errexit # Exit on error
set -o nounset # Don't use undefined variables
set -o xtrace # Print each command before execution

python train.py experiment=bert_mnli_debias_pruned datamodule.batch_size=256 trainer.max_epochs=3
python train.py experiment=bert_mnli_debias_pruned datamodule.batch_size=256 trainer.max_epochs=3 model.freeze_weights=True
python train.py experiment=bert_mnli_debias_pruned datamodule.batch_size=32 trainer.max_epochs=12
python train.py experiment=bert_mnli_debias_pruned datamodule.batch_size=32 trainer.max_epochs=12 model.freeze_weights=True