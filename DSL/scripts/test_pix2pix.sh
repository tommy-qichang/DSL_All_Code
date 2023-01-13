set -ex
#python test.py --dataroot ./datasets/stroke_ct --name stroke_ct --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode aligned --norm batch

#python train.py --dataroot ./datasets/stroke_ct --name stroke_ct --model pix2pix --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --continue_train

#python test.py --dataroot /share_hd1/db/BRATS/2018 --name brats --model pix2pix --netG unet_256 --direction AtoB --dataset_mode brats --norm instance --epoch 100 --results_dir results/brats/vis/epoch100

 #python test.py --dataroot /share_hd1/db/BRATS/2018 --name brats --model pix2pix --netG unet_256 --direction AtoB --dataset_mode brats --norm instance

#python test.py --dataroot /share_hd1/db/BRATS/2018 --name brats_resnet --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats --norm instance --epoch 50 --results_dir results/brats_resnet/vis/epoch50


#python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel --model pix2pix --netG unet_256 --direction AtoB --dataset_mode brats --norm instance --epoch 50 --results_dir results/brats_multilabel
#python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel --model pix2pix --netG unet_256 --direction AtoB --dataset_mode brats --norm instance --epoch 100 --results_dir results/brats_multilabel
#python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel --model pix2pix --netG unet_256 --direction AtoB --dataset_mode brats --norm instance --epoch 150 --results_dir results/brats_multilabel
#python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel --model pix2pix --netG unet_256 --direction AtoB --dataset_mode brats --norm instance --epoch 200 --results_dir results/brats_multilabel
#
##python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel_resnet --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats --epoch 30 --results_dir results/brats_multilabel
#python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel_resnet --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats --epoch 50 --results_dir results/brats_multilabel
##python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel_resnet --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats --epoch 70 --results_dir results/brats_multilabel
#python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel_resnet --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats --epoch 100 --results_dir results/brats_multilabel
#python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel_resnet --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats --epoch 150 --results_dir results/brats_multilabel



#python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_dadgan --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats --epoch 65 --results_dir results/brats_dadgan
#python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_dadgan_verticle_flip --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats --epoch 50 --results_dir results/brats_dadgan_flip

#python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_multilabel_resnet_perceptual_loss --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats --epoch 65 --results_dir results/brats_multilabel_resnet_perceptual_loss_da

#python save_syn.py --dataroot /share_hd1/db/BRATS/2018 --name brats_gan_hgglgg_perceptionloss100 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats_hgglgg --epoch 85 --results_dir results/brats_gan_perception100_hgglgg

#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/ --name brats_gan_nuclei --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --epoch 300 --results_dir results/nuclei_3_512 --load_size 512 --crop_size 512


#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/for_seg --name brats_gan_nuclei_withoutL1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --epoch 230 --results_dir results/nuclei_withoutL1


#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/for_seg --name nuclei_dadgan_withoutL1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --epoch 300 --results_dir results/nuclei_dadgan_withoutL1

#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/lifelong/256 --name lifelong_nuclei_finetune_D1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --subset 1 --results_dir results/lifelong_nuclei_finetune_D1

#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/lifelong/256 --name lifelong_nuclei_all --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --subset 0 --results_dir results/lifelong_nuclei_all

#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm --name lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --subset 1 --norm instance  --results_dir results/lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D2_breast

#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm --name lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D2 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --subset 2 --norm instance  --results_dir results/lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D2_kidney

#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm --name lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D3 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --subset 1 --norm instance  --results_dir results/lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D3_breast
#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm --name lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D3 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --subset 2 --norm instance  --results_dir results/lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D3_kidney
#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm --name lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D3 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --subset 3 --norm instance  --results_dir results/lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D3_liver

#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm --name lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D4 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --subset 1 --norm instance  --results_dir results/lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D4_breast
#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm --name lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D4 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --subset 2 --norm instance  --results_dir results/lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D4_kidney
#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm --name lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D4 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --subset 3 --norm instance  --results_dir results/lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D4_liver
#python save_syn_nuclei.py --dataroot /share_hd1/db/Nuclei/lifelong/256_no_color_norm --name lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D4 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --subset 4 --norm instance  --results_dir results/lifelong_nuclei_finetune_multiorgan_nocolornorm_instancenorm_D4_prostate

#python save_syn.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_1ch --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats1db --epoch 200 --results_dir results/brats_AsynDGANv2_1db_exp9_fixbug

#python save_syn.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats1db --epoch 100 --results_dir results/brats_AsynDGANv2_3db_exp10

#python save_syn.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_9d --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats1db --epoch 140 --results_dir results/brats_AsynDGANv2_3db_exp11

#python save_syn.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_doubleemod --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats1db --epoch latest --results_dir results/brats_AsynDGANv2_3db_exp33



#python save_syn.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_random_split_1ch --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats1db --epoch 200 --results_dir results/brats_AsynDGANv2_random_split_1ch_exp13_fixbug

#python save_syn.py --dataroot /freespace/local/qc58/dataset/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_singlemod --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats1db --epoch 200 --results_dir results/brats_AsynDGANv2_exp20_new3ch_singlemod

#python save_syn.py --dataroot /research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_doubleemod_t1c --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats1db --epoch 200 --results_dir results/brats_AsynDGANv2_3db_exp43_2_doubleemod_t1c

python save_syn.py --dataroot /research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_3mod_weighted --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats1db --epoch 200 --results_dir results/brats_AsynDGANv2_3db_3ch_exp25_v6

python save_syn.py --dataroot /research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2 --name Brats3db_resnet_3ch_9d --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats1db --epoch 200 --results_dir results/brats_AsynDGANv2_3db_exp11_v0


python save_syn.py --dataroot /research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2 --name Brats3db_resnet_random_split_1ch --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats1db --epoch 200 --results_dir results/brats_AsynDGANv2_random_split_1ch_exp13_fixbug

#python save_syn_memory_gan.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name memory_gan_t2_orig --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --dataset_mode brats_hgglgg --epoch 100  --norm instance --results_dir results
python save_syn_memory_gan.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name memory_gan_t1 --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --dataset_mode brats_hgglgg --epoch 25  --norm instance --results_dir results  --memory_gan t1
#python save_syn_memory_gan.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name memory_gan_t1 --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --dataset_mode brats_hgglgg --epoch 200  --norm instance --results_dir results
#
#python save_syn_memory_gan.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name memory_gan_ --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --dataset_mode brats_hgglgg --epoch 60 --results_dir results/memory_gan_t2_orig

python save_syn_memory_gan.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name memory_gan_t1 --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --dataset_mode brats_hgglgg --batch_size 10  --epoch 25 --gpu_ids 3 --memory_gan t1  --norm instance --results_dir results

#python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name memory_gan --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --lambda_L1 100 --dataset_mode brats_hgglgg --pool_size 0 --gpu_ids 7 --batch_size 10 --num_threads 0 --norm instance --memory_gan t1ce --epoch 60
#python save_syn.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name memory_gan --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --lambda_L1 100 --dataset_mode brats_hgglgg --pool_size 0 --gpu_ids 0 --batch_size 10 --num_threads 0 --norm instance --memory_gan t1ce --epoch 60

#python save_syn.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name LGG_pretrain_modalitybank --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --lambda_L1 100 --dataset_mode brats_hgglgg --pool_size 0 --gpu_ids 0 --batch_size 10 --num_threads 0 --norm instance --memory_gan t1ce --epoch 60
python save_syn.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name LGG_pretrain_modalitybank --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats_lgg --gpu_ids 4 --batch_size 20 --epoch 200

python save_syn_memory_gan.py --dataroot /research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2 --name memory_gan_rebuttal_lgg_pretrain/ --model pix2pix --netG adafm_resnet_9blocks --netD adafm --direction AtoB --dataset_mode brats_hgglgg --epoch 35 --memory_gan t1  --norm instance  --results_dir results/memory_gan_rebuttal_lgg_pretrain/

python save_syn_memory_gan.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name memory_gan_rebuttal_lgg_pretrain/ --model pix2pix --netG adaconv_resnet_9blocks --netD adaconv --direction AtoB --dataset_mode brats_hgglgg --epoch 35 --memory_gan t1,t2,flair  --norm instance  --results_dir results

python save_syn_memory_gan.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/processed --name memory_gan_rebuttal_mms_pretrain --model pix2pix --netG adaconv_resnet_9blocks --netD adaconv --direction AtoB --dataset_mode brats_hgglgg --epoch 45 --memory_gan t1,t2,flair  --norm instance  --results_dir results


#python train.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name memory_gan_rebuttal_mms_pretrain --model dadgan3ch --netG adaconv_resnet_9blocks --netD adaconv --direction AtoB --lambda_L1 100 --dataset_mode brats3db --pool_size 0 --gpu_ids 5 --batch_size 8 --num_threads 0 --norm instance --memory_gan t1,t2,flair --epoch 200 --d_size 3
#exp2
python save_syn_memory_gan.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name memory_gan_rebuttal_lgg_pretrain --model pix2pix --netG adaconv_resnet_9blocks --netD adaconv --direction AtoB --dataset_mode brats1db --epoch 90 --norm instance --results_dir results --memory_gan t1,t2,flair

python save_syn_memory_gan.py --dataroot /research/cbim/vast/qc58/pub-db/BraTS2018/AsynDGANv2 --name memory_gan_rebuttal_mms_pretrain --model pix2pix --netG adaconv_resnet_9blocks --netD adaconv --direction AtoB --dataset_mode brats1db --epoch 55 --norm instance --results_dir results --memory_gan t1,t2,flair


#def list_shape(url, type):
#  f = h5py.File(url,"r")
#  for i in list(f[f'{type}']):
#    data = f[f'{type}/{i}/data'][()]
#    label = f[f'{type}/{i}/label'][()]
#    print(f"{data.shape}|{label.shape}")

#for i in list(f['test']):
#  label = f[f"test/{i}/label"][()]
#  print(np.unique(label))




