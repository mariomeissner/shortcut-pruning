#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh

set -ex # Print commands as they run, exit if failure

python train.py experiment=bert_mnli_pruned datamodule.batch_size=256 trainer.max_epochs=3
python train.py experiment=bert_mnli_pruned datamodule.batch_size=32 trainer.max_epochs=3
python train.py experiment=bert_mnli_pruned datamodule.batch_size=32 trainer.max_epochs=12
python train.py experiment=bert_mnli_pruned datamodule.batch_size=256 trainer.max_epochs=3 model.freeze_weights=True
python train.py experiment=bert_mnli_pruned datamodule.batch_size=32 trainer.max_epochs=12 model.freeze_weights=True