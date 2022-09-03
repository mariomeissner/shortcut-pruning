for i in {0..4} 
do
    echo "Running i=$i"
    printnum=`expr $i + 1`
    python scripts/extract_preds.py \
    /PATH.../checkpoints/last.ckpt /
    /PATH.../undertrain-mnli-1$i.json \
    --dataset_name mnli
done
echo "Finished batch script!"