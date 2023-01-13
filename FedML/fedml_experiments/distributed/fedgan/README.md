## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments

## BRATS generation experiments
3 clients training, multi-modality data, 200 rounds, 1 epoch per round:
```
sh run_fedgan_distributed_pytorch.sh 4 exp_4.yml
```


##run on background
```
nohup sh run_fedgan_distributed_pytorch.sh 4 exp_1.yml> ./log-fedgan-exp1.txt 2>&1 &
nohup sh run_fedgan_distributed_pytorch.sh 5 exp_2.yml> ./log-fedgan-exp2.txt 2>&1 &
nohup sh run_fedgan_distributed_pytorch.sh 4 exp_4.yml> ./log-fedgan-exp4.txt 2>&1 &
```


## save synthetic images
save for downstream task:
```
python save_syn.py --cfg exp_1.yml --batch_size 1 --save_dir ./run/heart/dadgan/experiment_1 --epoch 200 --GPUid 0 --save_data
python save_syn.py --cfg exp_2.yml --batch_size 1 --save_dir ./run/path/dadgan/experiment_0 --epoch 200 --GPUid 0 --save_data
python save_syn.py --cfg exp_4.yml --batch_size 1 --save_dir ./run/brats/dadgan_mc/experiment_0 --epoch 200 --GPUid 0 --save_data
```

print only:
```
python save_syn.py --cfg exp_1.yml --batch_size 20 --save_dir ./run/heart/dadgan/experiment_1 --epoch 200 --GPUid 0 --num_test 3
python save_syn.py --cfg exp_2.yml --batch_size 20 --save_dir ./run/path/dadgan/experiment_0 --epoch 200 --GPUid 0 --num_test 3
python save_syn.py --cfg exp_4.yml --batch_size 20 --save_dir ./run/brats/dadgan_mc/experiment_0 --epoch 200 --GPUid 0 --num_test 3
```
