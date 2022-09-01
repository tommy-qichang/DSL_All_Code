#!/bin/sh

#i=80
#while [ "$i" -le 200 ]; do
#    python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch $i --gpu_ids 1  --results_dir results/fsl_4ch_exp1_modality_nature_new
#    i=$(( i + 10 ))
#done

#i=200
#while [ "$i" -ge 90 ]; do
#    python save_syn_fsl.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_cardiac_exp1_nature --input_nc 1 --output_nc 1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch $i --gpu_ids 3  --results_dir results/fsl_cardiac_exp1_nature
#    i=$(( i - 10 ))
#done

#i=80
#while [ "$i" -ge 0 ]; do
#    python save_syn_fsl.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_cardiac_exp1_nature --input_nc 1 --output_nc 1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch $i --gpu_ids 3  --results_dir results/fsl_cardiac_exp1_nature
#    i=$(( i - 10 ))
#done
#
#i=800
#while [ "$i" -ge 0 ]; do
#    python save_syn_nuclei.py --dataroot /research/cbim/vast/qc58/pub-db/Hist/exp2_path/for_gan_training_286_no_color_norm/  --name hist_l1300 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --epoch $i --gpu_ids 3  --results_dir results/fsl_hist_l1300_no_color_norm
#    i=$(( i - 10 ))
#done

#i=400
#while [ "$i" -ge 10 ]; do
#    python save_syn_nuclei.py --dataroot /research/cbim/vast/qc58/pub-db/Hist/exp2_path/for_gan_training_286/  --name hist_l1300_norm_verify --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --epoch $i --gpu_ids 7  --results_dir results/hist_l1300_norm_verify
#    i=$(( i - 10 ))
#done

#i=190
#while [ "$i" -ge 10 ]; do
#    python save_syn.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2  --name Brats4ch3db_triplemod --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats_4ch  --input_nc 4 --output_nc 4 --epoch $i --gpu_ids 7  --results_dir results/Brats4ch3db_triplemod
#    i=$(( i - 10 ))
#done

python save_syn.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name Brats4ch3db_modality_nature --input_nc 4 --output_nc 4 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats_4ch --epoch 160 --load_size 256 --crop_size 256 --gpu_ids 1  --results_dir results/Brats4ch3db_modality_nature_fid_exp4_6_da_1

python save_syn.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name Brats4ch3db_modality_nature --input_nc 4 --output_nc 4 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats_4ch --epoch 160 --gpu_ids 1  --results_dir results/Brats4ch3db_modality_nature_fid_exp4_6_2

python save_syn.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name Brats4ch3db_modality_nature --input_nc 4 --output_nc 4 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats_4ch --epoch 160 --gpu_ids 1  --results_dir results/Brats4ch3db_modality_nature_fid_exp4_6_3

python save_syn_fsl.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name CTA3db_nature_fid --input_nc 1 --output_nc 1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 200 --load_size 256 --crop_size 256 --gpu_ids 1 --results_dir results/CTA3db_nature_fid_exp1-6_da_3

#python save_syn_fsl.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name CTA3db_nature_fid --input_nc 1 --output_nc 1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 160 --gpu_ids 1 --results_dir results/CTA3db_nature_fid_exp1-6_2
#
#python save_syn_fsl.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name CTA3db_nature_fid --input_nc 1 --output_nc 1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 160 --gpu_ids 1 --results_dir results/CTA3db_nature_fid_exp1-6_3
#
##python save_syn_fsl.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name CTA3db_nature_fid --input_nc 1 --output_nc 1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 160 --gpu_ids 1 --results_dir results/CTA3db_nature_fid_exp1-6_4
#
#python save_syn_nuclei.py --dataroot /research/cbim/vast/qc58/pub-db/Hist/exp2_path/for_gan_training_286/  --name hist_l1300_norm --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --epoch 190 --gpu_ids 1  --results_dir results/hist_l1300_norm_2_2
#
#python save_syn_nuclei.py --dataroot /research/cbim/vast/qc58/pub-db/Hist/exp2_path/for_gan_training_286/  --name hist_l1300_norm --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --epoch 190 --gpu_ids 1  --results_dir results/hist_l1300_norm_2_3

#python save_syn_nuclei.py --dataroot /research/cbim/vast/qc58/pub-db/Hist/exp2_path/for_gan_training_286/  --name hist_l1300_norm --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --epoch 190 --gpu_ids 1  --results_dir results/hist_l1300_norm_2

#nohup python train.py --dataroot /research/cbim/vast/qc58/pub-db/Hist/exp2_path/for_gan_training_286/ --name hist_l1300_norm --model dadgan --netG resnet_9blocks --direction AtoB --lambda_L1 100 --delta_perceptual 10  --dataset_mode nuclei_split --pool_size 0  --gpu_ids 1 --batch_size 8 --norm instance --num_threads 0 --niter=600 --niter_decay=400 --d_size 4

#new exp1
#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 200 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new
#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 190 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new
#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 180 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new
#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 170 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new
#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 160 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new
#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 150 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new
#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 140 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new
#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 130 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new
#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 120 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new
#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 110 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new
#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 100 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new
#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 90 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new
