## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments


## BRATS segmentation experiments

3 clients, T1&T2&Flair&T1c, 200 rounds, 1 epoch per round:
```
nohup sh run_fedseg_distributed_pytorch.sh 4 exp_1.yml> ./log-fedseg-exp1.txt 2>&1 &
nohup sh run_fedseg_distributed_pytorch.sh 5 exp_2.yml> ./log-fedseg-exp2.txt 2>&1 &
nohup sh run_fedseg_distributed_pytorch.sh 4 exp_4.yml> ./log-fedseg-exp4.txt 2>&1 &
nohup sh run_fedseg_distributed_pytorch.sh 4 exp_4_miss_mod.yml> ./log-fedseg-exp4_mm.txt 2>&1 &
```