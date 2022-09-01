set -ex
#python train.py --dataroot ./datasets/edges2handbags --name edges2handbags --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
#python train.py --dataroot ./datasets/stroke_ct --name stroke_ct --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --continue_train

#python train.py --dataroot /share_hd1/db/BRATS/brats_p2p --name brats --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --pool_size 0 --gpu_ids 2 --batch_size 16

#python train.py --dataroot /share_hd1/db/BRATS/brats_p2p --name brats --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --pool_size 0 --gpu_ids 2 --batch_size 32 --norm instance

#python train.py --dataroot /share_hd1/db/BRATS/brats_p2p --name brats_resnet --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode aligned --pool_size 0 --gpu_ids 3 --batch_size 16 --norm instance

#python train.py --dataroot /share_hd1/db/BRATS/brats_p2p_random0 --name brats_random0_unet --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --pool_size 0 --gpu_ids 3 --batch_size 16

#python train.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode brats --pool_size 0 --gpu_ids 1 --batch_size 16 --norm instance

#python train.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel_resnet_keeptumor --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats --pool_size 0 --gpu_ids 1 --batch_size 10
#python train.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel_resnet_keepskull --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats --pool_size 0 --gpu_ids 3 --batch_size 10

#python train.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel_resnet_perceptual_loss --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats --pool_size 0 --gpu_ids 3 --batch_size 10

#python train.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel_resnet_perceptual_loss --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats --pool_size 0 --gpu_ids 3 --batch_size 10

#python train.py --dataroot /share_hd1/db/BRATS/2018/tumor_size_split_10 --name brats_dadgan --model dadgan --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats_split --pool_size 0 --gpu_ids 3 --batch_size 1 --num_threads 0

#python train.py --dataroot /share_hd1/db/BRATS/2018/tumor_size_split_10 --name brats_dadgan_verticle_flip --model dadgan --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats_split --pool_size 0 --gpu_ids 1 --batch_size 1 --num_threads 0

#python train.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel_resnet --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats --pool_size 0 --gpu_ids 3 --batch_size 10

#python train.py --dataroot /share_hd1/db/Brain/real_images --name Isles_multilabel_resnet --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode isles --pool_size 0 --gpu_ids 3 --batch_size 10

#python train.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_1ch --model dadgan --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 5 --batch_size 23 --d_size 3
python train.py --dataroot /research/cbim/vast/qc58/pub-db/ISLES2017/processed --name Isles2017_modalitybank --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode isles --pool_size 0 --gpu_ids 4 --batch_size 23

python train.py --dataroot /research/cbim/vast/qc58/pub-db/ACDC --name acdc_modalitybank --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode acdc --pool_size 0 --gpu_ids 5 --batch_size 10  --display_freq 300

python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name LGG_pretrain_modalitybank --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats_lgg --pool_size 0 --gpu_ids 4 --batch_size 20
#python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name LGG_pretrain_modalitybank2 --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats_lgg --pool_size 0 --gpu_ids 4 --batch_size 20  --continue_train

python train.py --dataroot /research/cbim/vast/qc58/pub-db/MMs/processing --name mms_pretrain_modalitybank2 --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode mms --pool_size 0 --gpu_ids 5 --batch_size 40 --continue_train


#--continue_train --epoch_count 140

#python train.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_9d_v2 --model dadgan3ch --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 6 --batch_size 10  --d_size 3
#python train.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_9d_v4 --model dadgan3ch --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 5 --batch_size 10 --update_html_freq 500  --d_size 3

#python train.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_random_split_1ch --model dadgan  --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 5 --batch_size 5  --d_size 10

#python train.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_singlemod --model dadgan3ch_singlemod --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 6 --batch_size 23 --update_html_freq 500 --display_ncols 7 --d_size 3

#python train.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_doubleemod --model dadgan3ch_doublemod --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 5 --batch_size 12 --update_html_freq 500 --display_ncols 7 --d_size 3

#python train.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_doubleemod_t1c --model dadgan3ch_doublemod --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 5 --batch_size 12 --update_html_freq 500 --display_ncols 7 --d_size 3
#
#python train.py --dataroot /research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_3mod_weighted --model dadgan3ch --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 5 --batch_size 12 --update_html_freq 500 --display_ncols 7 --d_size 3
#
#python train.py --dataroot /research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_3mod_weighted_t1 --model dadgan3ch --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 4 --batch_size 12 --update_html_freq 2000 --display_ncols 7 --d_size 3

#python train.py --dataroot /research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_all --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats1db --pool_size 0 --gpu_ids 6 --batch_size 16
#python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name memory_gan --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --lambda_L1 100 --dataset_mode brats_hgglgg --pool_size 0 --gpu_ids 7 --batch_size 10 --num_threads 0 --norm instance --memory_gan


#python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name memory_gan --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --lambda_L1 100 --dataset_mode brats_hgglgg --pool_size 0 --gpu_ids 7 --batch_size 10 --num_threads 0 --norm instance --memory_gan t1ce --epoch 60

python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name memory_gan_t1ce --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --lambda_L1 100 --dataset_mode brats_hgglgg --pool_size 0 --gpu_ids 7 --batch_size 10 --num_threads 0 --norm instance --memory_gan t1ce --continue_train --epoch 110


python train.py --dataroot /share_hd1/db/BRATS/2018 --name memory_gan --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --lambda_L1 100 --dataset_mode brats_hgglgg --pool_size 0 --gpu_ids 1 --batch_size 10 --num_threads 0 --norm instance --memory_gan t1 --epoch 60


python train.py --dataroot /research/cbim/vast/qc58/pub-db/ISLES2017/processed --name memory_gan_isles --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --lambda_L1 100 --dataset_mode isles --pool_size 0 --gpu_ids 6 --batch_size 10 --num_threads 0 --norm instance --memory_gan


python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name memory_gan_exp2_2_not1c --model dadgan3ch --netG adaconv_resnet_9blocks --netD adaconv --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 4 --batch_size 8 --num_threads 0 --norm instance --memory_gan t1,t2,flair --epoch 90 --d_size 3


python train.py --dataroot /research/cbim/vast/qc58/pub-db/brain_mri/rsna-intracranial-hemorrhage-detection/processed --name memory_gan_brainct --model pix2pix --netG adaconv_resnet_9blocks --netD adaconv --direction AtoB --lambda_L1 100 --dataset_mode brainct --pool_size 0 --gpu_ids 6 --batch_size 10 --num_threads 0 --norm instance

#python train.py --dataroot /share_hd1/db/BRATS/brats_p2p_random0 --name brats_random0_unet --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --pool_size 0 --gpu_ids 3 --batch_size 16

python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name Brats3db_resnet_t2_fedml --model dadgan --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 5 --batch_size 20 --update_html_freq 500  --num_threads 0 --norm instance --d_size 3 --load_size 256 --crop_size 256 --no_flip

python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name Brats4ch3db_modality_nature --input_nc 4 --output_nc 4 --model dadgan --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats_4ch3db --pool_size 0 --gpu_ids 3 --batch_size 10 --display_freq 300  --num_threads 0 --norm instance --d_size 3


python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name memory_gan_rebuttal_lgg_pretrain --model dadgan3ch --netG adaconv_resnet_9blocks --netD adaconv --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 6 --batch_size 8 --num_threads 0 --norm instance --memory_gan t1,t2,flair --epoch 200 --d_size 3

python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name memory_gan_rebuttal_mms_pretrain --model dadgan3ch --netG adaconv_resnet_9blocks --netD adaconv --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 5 --batch_size 8 --num_threads 0 --norm instance --memory_gan t1,t2,flair --epoch 200 --d_size 3


python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name Brats4ch3db_modality_nature --input_nc 4 --output_nc 4 --model dadgan4ch --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats_4ch3db --pool_size 0 --gpu_ids 4 --batch_size 10 --display_freq 300  --num_threads 0 --norm instance --d_size 3

nohup python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name Brats4ch3db_modality_nature_fid_faster --input_nc 4 --output_nc 4 --model dadgan4ch --netG resnet_9blocks --direction AtoB --lambda_L1 150 --dataset_mode brats_4ch3db --pool_size 0 --gpu_ids 0 --batch_size 10 --num_threads 0 --norm instance --d_size 3

nohup python train.py --dataroot /research/cbim/medical/medical-share/public/MMWHS/FSL --name CTA3db_nature_fid --input_nc 1 --output_nc 1 --model fsl3db --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode fsl3db --pool_size 0 --gpu_ids 1 --batch_size 10 --display_freq 300 --num_threads 0 --norm instance --d_size 3



