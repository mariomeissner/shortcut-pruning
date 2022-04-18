for i in {0..4} 
do
    echo "Running i=$i"
    printnum=`expr $i + 1`
    python extract_preds.py \
    logs/experiments/confidence_exp/multiruns/2022-04-18/14-21-01/$i/checkpoints/last.ckpt \
    data/weak_models/undertrained/mnli/confidence_exp/epoch-$printnum.json \
    --dataset_name mnli
done
echo "Finished batch script!"