python train.py --dataroot /research/cbim/vast/qc58/pub-db/MMWHS/processed/fsl/2d --name FSL_exp1 --input_nc 3 --output_nc 2 --model fsl --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode fsl3db --pool_size 0 --gpu_ids 4 --batch_size 10 --display_freq 300  --num_threads 0 --norm instance --d_size 3


python train.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model fsl --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode fsl4db --pool_size 0 --gpu_ids 5 --batch_size 10 --display_freq 300  --num_threads 0 --norm instance --d_size 4

#Exp46~48
python train.py --dataroot /research/cbim/vast/qc58/local/local_db/tmp_data --name Brats3db_resnet_3ch_doubleemod --model dadgan3ch_doublemod --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 1 --batch_size 12 --update_html_freq 500 --display_ncols 7 --d_size 3

#Exp4-20 4ch missing modality
python train.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name Brats4ch_exp4-20_triplemod --input_nc 4 --output_nc 2 --model fsl --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode fsl4db --pool_size 0 --gpu_ids 5 --batch_size 10 --display_freq 300  --num_threads 0 --norm instance --d_size 4

nohup python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name Brats4ch3db_triplemod --input_nc 4 --output_nc 4 --model dadgan4ch_triplemod --netG resnet_9blocks --direction AtoB --lambda_L1 150 --dataset_mode brats_4ch3db --pool_size 0 --gpu_ids 3 --batch_size 10 --num_threads 0 --norm instance --d_size 3

