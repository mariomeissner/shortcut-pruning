for i in {0..4} 
do
    echo "Running i=$i"
    printnum=`expr $i + 1`
    python extract_preds.py \
    /remote/csifs1/disk0/meissner/shortcut-pruning/experiments/weak/undertrain/bert/mnli/multiruns/2022-04-18/16-09-54/$i/checkpoints/last.ckpt /
    data/weak_models/undertrained/mnli/undertrain-mnli-1$i.json \
    --dataset_name mnli
done
echo "Finished batch script!"