for i in {1..4} 
do
    python train.py experiment=debias/bert_squad name=debiasing/tinybert/squad \
    seed=1$i datamodule.bias_path=/home/meissner/shortcut-pruning/data/weak_models/tinybert/squad/tiny-squad-1$i.json trainer.gpus='[2]'
done
echo "Finished batch script!"