## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments

## AsynDGAN on BRATS experiments
3 clients training, T2 data, 200 epochs, batch size 20/client (sample method 'balance'):
```
sh run_asdgan_distributed_pytorch.sh 4 default.yml
```


## run on background
```
nohup sh run_asdgan_distributed_pytorch.sh 4 default.yml > ./log_default.txt 2>&1 &
nohup sh run_asdgan_distributed_pytorch.sh 4 n_exp_1.yml> ./log_nature_exp1.txt 2>&1 &
nohup sh run_asdgan_distributed_pytorch.sh 5 n_exp_2.yml> ./log_nature_exp2.txt 2>&1 &
nohup sh run_asdgan_distributed_pytorch.sh 4 n_exp_4.yml> ./log_nature_exp4.txt 2>&1 &
nohup sh run_asdgan_distributed_pytorch.sh 4 n_exp_4_miss_mod.yml> ./log_nature_exp4_mm.txt 2>&1 &
```


## save synthetic images
save 3 * 20 synthetic images to visualize
```
python save_syn.py --cfg default.yml --batch_size 20 --save_dir ./run/brats_t2/asdgan/experiment_1 --epoch 50 --GPUid 0 --num_test 3

python save_syn.py --cfg n_exp_1.yml --batch_size 20 --save_dir ./run/heart/asdgan/experiment_0 --epoch 200 --GPUid 0 --num_test 3
python save_syn.py --cfg n_exp_2.yml --batch_size 20 --save_dir ./run/path/asdgan/experiment_0 --epoch 200 --GPUid 0 --num_test 3
python save_syn.py --cfg n_exp_4.yml --batch_size 20 --save_dir ./run/brats/asdgan_mc/experiment_0 --epoch 200 --GPUid 0 --num_test 3
```


save all synthetic images to h5 file for training segmentation model
```
python save_syn.py --cfg exp_2.yml --batch_size 20 --save_dir ./run/brats_t2/asdgan/experiment_2 --epoch 200 --GPUid 0 --num_test -1 --save_data

python save_syn.py --cfg exp_5.yml --batch_size 20 --save_dir ./run/brats_t2/asdgan/experiment_5 --epoch 200 --GPUid 0 --num_test -1 --save_data

python save_syn.py --cfg n_exp_1.yml --batch_size 1 --save_dir ./run/heart/asdgan/experiment_0 --epoch 200 --GPUid 0 --save_data
python save_syn.py --cfg n_exp_2.yml --batch_size 1 --save_dir ./run/path/asdgan/experiment_0 --epoch 200 --GPUid 0 --save_data
python save_syn.py --cfg n_exp_4.yml --batch_size 1 --save_dir ./run/brats/asdgan_mc/experiment_0 --epoch 200 --GPUid 0 --save_data
```
