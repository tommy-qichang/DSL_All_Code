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

#python train.py --dataroot /share_hd1/db/Nuclei --name brats_gan_nuclei_withoutL1 --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 0 --delta_perceptual 10 --dataset_mode nuclei --pool_size 0 --gpu_ids 2 --batch_size 12 --num_threads 0 --continue_train --niter=200 --niter_decay=200

#python train.py --dataroot /share_hd1/db/Nuclei/for_seg --name nuclei_dadgan_withoutL1 --model dadgan --netG resnet_9blocks --direction AtoB --lambda_L1 0 --delta_perceptual 10 --dataset_mode nuclei_split --pool_size 0 --gpu_ids 1 --batch_size 3 --num_threads 0 --niter=200 --niter_decay=200

#python train.py --dataroot /share_hd1/db/Nuclei/lifelong/256 --name lifelong_nuclei_all --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --delta_perceptual 10 --dataset_mode nuclei --pool_size 0 --gpu_ids 2 --batch_size 12 --num_threads 0 --continue_train --niter=200 --niter_decay=200

#Train lifelone of dataset1:
#python train.py --dataroot /share_hd1/db/Nuclei/lifelong/256 --name lifelong_nuclei_finetune_D1 --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --delta_perceptual 10  --dataset_mode nuclei --pool_size 0  --subset 1 --gpu_ids 3 --batch_size 8  --num_threads 0 --niter=200 --niter_decay=200
#python train.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm/ --name lifelong_nuclei_finetune_multiorgan_nocolornorm_D1 --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --delta_perceptual 10  --dataset_mode nuclei --pool_size 0  --subset 1 --gpu_ids 0 --batch_size 8  --num_threads 0 --niter=200 --niter_decay=200


#Train lifelone of dataset2:
#python train.py --dataroot /share_hd1/db/Nuclei/lifelong/256 --name lifelong_nuclei_finetune_D2 --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --delta_perceptual 10  --dataset_mode nuclei --pool_size 0  --subset 2 --gpu_ids 3 --batch_size 8 --continue_train  --num_threads 0 --niter=200 --niter_decay=200

#python train.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm/ --name lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D2 --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --delta_perceptual 10  --dataset_mode nuclei --pool_size 0  --subset 2 --gpu_ids 0 --batch_size 8 --epoch 400 --norm instance --continue_train  --num_threads 0 --niter=200 --niter_decay=200

#python train.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm/ --name lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D3 --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --delta_perceptual 10  --dataset_mode nuclei --pool_size 0  --subset 3 --gpu_ids 0 --batch_size 8 --norm instance --continue_train  --num_threads 0 --niter=200 --niter_decay=200

python train.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm/ --name lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D4 --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --delta_perceptual 10  --dataset_mode nuclei --pool_size 0  --subset 4 --gpu_ids 0 --batch_size 8 --norm instance --continue_train  --num_threads 0 --niter=200 --niter_decay=200


#Train lifelone of dataset2 with no color norm
#python train.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm/ --name lifelong_nuclei_finetune_multiorgan_nocolornorm_D2 --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --delta_perceptual 10  --dataset_mode nuclei --pool_size 0  --subset 2 --gpu_ids 1 --batch_size 8 --continue_train --num_threads 0 --niter=200 --niter_decay=200

nohup python train.py --dataroot /research/cbim/vast/qc58/pub-db/Hist/exp2_path/for_gan_training_286_no_color_norm/ --name hist_l1300 --model dadgan --netG resnet_9blocks --direction AtoB --lambda_L1 300 --delta_perceptual 10  --dataset_mode nuclei_split --pool_size 0  --gpu_ids 2 --batch_size 8 --norm instance --num_threads 0 --niter=200 --niter_decay=200 --d_size 4

nohup python train.py --dataroot /research/cbim/vast/qc58/pub-db/Hist/exp2_path/for_gan_training_286_no_color_norm/ --name hist_l1300 --model dadgan --netG resnet_9blocks --direction AtoB --lambda_L1 300 --delta_perceptual 10  --dataset_mode nuclei_split --pool_size 0  --gpu_ids 2 --batch_size 8 --norm instance --num_threads 0 --niter=400 --niter_decay=400 --d_size 4  --continue_train ---epoch_count 400

nohup python train.py --dataroot /research/cbim/vast/qc58/pub-db/Hist/exp2_path/for_gan_training_286/ --name hist_l1300_norm_verify --model dadgan --netG resnet_9blocks --direction AtoB --lambda_L1 100 --delta_perceptual 10  --dataset_mode nuclei_split --pool_size 0  --gpu_ids 5 --batch_size 8 --norm instance --num_threads 0 --niter=200 --niter_decay=200 --d_size 4


