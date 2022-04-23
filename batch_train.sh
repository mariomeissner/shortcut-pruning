for i in {1..5} 
do
    python train.py experiment=debias/bert_mnli name=confidence_exp \
    datamodule.bias_path=/home/meissner/shortcut-pruning/data/weak_models/undertrained/mnli/confidence_exp/epoch-$i.json
done
echo "Finished batch script!"