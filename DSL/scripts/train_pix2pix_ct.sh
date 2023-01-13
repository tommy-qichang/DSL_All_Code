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

python train.py --dataroot /share_hd1/db/ --name ct_multilabel_resnet_perceptual_loss --model pix2pix --continue_train --netG resnet_9blocks --direction AtoB --lambda_L1 100 --delta_perceptual 1 --dataset_mode ct --pool_size 0 --gpu_ids 2 --batch_size 14 --num_threads 0

#python train.py --dataroot /share_hd1/db/BRATS/2018 --name brats_dadgan_hgglgg --model dadgan --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats_hgglgg_split --pool_size 0 --gpu_ids 0 --batch_size 1 --num_threads 0



