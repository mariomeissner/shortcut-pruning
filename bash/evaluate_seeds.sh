#!/bin/bash
set -o errexit # Exit on error
set -o nounset # Don't use undefined variables
set -o xtrace # Print each command before execution

SEEDS=(000 111 222 333 444 555 666 777 888 999)
RESULT_FILE="seeds_results.txt"
touch $RESULT_FILE

# MNLI
echo "MNLI Validation Matched Results" >> $RESULT_FILE

# Evaluate vanilla bert
echo "Vanilla Results" >> $RESULT_FILE
for seed in ${SEEDS[@]}; do
    python evaluate_mnli.py logs/bert_mnli_seeds/seed_$seed/checkpoints/last.ckpt >> $RESULT_FILE
done

# Evaluate pruned bert
echo "Pruning results" >> $RESULT_FILE
for seed in ${SEEDS[@]}; do
    python evaluate_mnli.py logs/bert_mnli_pruned_seeds/seed_$seed/checkpoints/last.ckpt --pruned_model >> $RESULT_FILE
done
echo "" >> $RESULT_FILE

# HANS
echo "HANS Results" >> $RESULT_FILE

# Evaluate vanilla bert
echo "Vanilla Results" >> $RESULT_FILE
for seed in ${SEEDS[@]}; do
    python evaluate_hans.py logs/bert_mnli_seeds/seed_$seed/checkpoints/last.ckpt >> $RESULT_FILE
done

# Evaluate pruned bert
echo "Pruning results" >> $RESULT_FILE
for seed in ${SEEDS[@]}; do
    python evaluate_hans.py logs/bert_mnli_pruned_seeds/seed_$seed/checkpoints/last.ckpt --pruned_model >> $RESULT_FILE
done
