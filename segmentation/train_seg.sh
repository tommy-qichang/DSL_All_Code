#python train.py -c config_files/config_fsl_seg_exp1-fedgan-split.json -d 3


#python train.py -c config_files/config_BRATS_4ch_exp_4_train_val.json -d 0

#python train.py -c config_files/config_fsl_seg_exp1-fedgan.json -d 0

#python train.py -c config_files/config_fsl_seg_exp1-real.json -d 0

python train.py -c config_files/config_BRATS_4ch_exp_4_train_real_3d.json -d 0,1,2,3

#python train.py -c config_files/config_BRATS_4ch_exp_4_train_asdgan_split_3d.json -d 0,1,2,3
srun --partition=pat_mercury --gres=gpu:4 -N1 --ntasks-per-node=1 --quotatype=auto --job-name=nseg-train --kill-on-bad-exit=1 python train.py -c config_files/config_BRATS_4ch_exp_4_train_asdgan_split_3d.json -d 0,1,2,3 2>&1|tee log/train-exp4-seg-3d-160data_1209.log &

srun --partition=pat_mercury --gres=gpu:4 -N1 --ntasks-per-node=1 --quotatype=auto --job-name=nseg-train --kill-on-bad-exit=1 python train.py -c config_files/config_BRATS_4ch_exp_4_train_real_3d.json -d 0,1,2,3 2>&1|tee log/train-exp4-seg-3d-realdata_1207.log &

python train.py -c config_files/config_BRATS_4ch_exp_4_train_asdgan_augment_split.json -d 0,1,2,3

srun --partition=pat_mercury --gres=gpu:1 -N1 --ntasks-per-node=1 --quotatype=auto --job-name=nseg-train --kill-on-bad-exit=1 python train.py -c config_files/config_BRATS_4ch_exp_4_train_asdgan_split_160.json -d 3 2>&1|tee log/train-exp4-seg-160data.log &

python train.py -c config_files/config_BRATS_4ch_exp_4_train_asdgan_split_160.json -d 0,1,2,3

python train.py -c config_files/config_BRATS_4ch_exp_4_train_asdgan_split_160_2x.json -d 0,1,2,3

srun --partition=pat_mercury --gres=gpu:2 -N1 --ntasks-per-node=1 --quotatype=auto --job-name=nseg-train --kill-on-bad-exit=1 python train.py -c config_files/config_BRATS_4ch_exp_4_train_asdgan_split_160_2x.json -d 3,4 2>&1|tee log/train-exp4-seg-aug2x_1215.log &

python train.py -c config_files/config_fsl_seg_exp1-fedmedgan-split.json -d 1

python train.py -c config_files/config_BRATS_4ch_exp_4_train_fedmedgan.json -d 2,3

python train.py -c config_files/config_BRATS_4ch_exp_4_train_real_cbica.json -d 0,1,2,3