python save_syn.py --dataroot /research/cbim/vast/qc58/pub-db/MMWHS/processed/fsl/2d --name FSL_exp1 --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality mri --epoch 200 --results_dir results/FSL_exp1_acdc_mri

python save_syn.py --dataroot /research/cbim/vast/qc58/pub-db/MMWHS/processed/fsl/2d --name FSL_exp1 --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality mri2 --epoch 200 --results_dir results/FSL_exp1_whs_mri

python save_syn.py --dataroot /research/cbim/vast/qc58/pub-db/MMWHS/processed/fsl/2d --name FSL_exp1 --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 200 --results_dir results/FSL_exp1_whs_ct




python save_syn.py --dataroot /research/cbim/vast/qc58/pub-db/MMWHS/processed/fsl/2d --name FSL_exp1 --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 200 --results_dir results/FSL_exp1_whs_ct

#Epoch200
python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct1 --epoch 200 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature
python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct2 --epoch 200 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature
python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct3 --epoch 200 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature


#Epoch100
python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct1 --epoch 100 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature
python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct2 --epoch 100 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature
python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct3 --epoch 100 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature



python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality mri --epoch 200 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature
python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality mri --epoch 100 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new

python save_syn.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name Brats4ch3db_modality_nature_fid --input_nc 4 --output_nc 4 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats_4ch --epoch 85 --gpu_ids 1  --results_dir results/Brats4ch3db_modality_nature_fid_exp4_6

python save_syn_fsl.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name CTA3db_nature_fid --input_nc 1 --output_nc 1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 160 --gpu_ids 5 --results_dir results/CTA3db_nature_fid_exp1-6

#new exp1
python save_syn.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_4ch_exp1_modality_nature --input_nc 3 --output_nc 2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 200 --gpu_ids 5  --results_dir results/fsl_4ch_exp1_modality_nature_new

python save_syn_fsl.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name fsl_cardiac_exp1_nature --input_nc 1 --output_nc 1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 110 --gpu_ids 3  --results_dir results/fsl_cardiac_exp1_nature

python save_syn_nuclei.py --dataroot /research/cbim/vast/qc58/pub-db/Hist/exp2_path/for_gan_training_286/  --name hist_l1300_norm --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --epoch 190 --gpu_ids 7  --results_dir results/hist_l1300_norm_2



