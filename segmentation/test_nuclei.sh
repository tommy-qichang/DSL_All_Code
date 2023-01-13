
# results are saved in experiments/nuclei_seg/.../models/.../test_results/...

## fedgan
python test_nuclei_seg.py -c config_files/config_nuclei_seg_fedgan_server2.json -d 1 -r /data/zhennan/code_seg_cvpr/experiments/nuclei_seg/fedgan_unet/models/path_seg_fedgan/0824_103005/model_best.pth


## asdgan
#python test_nuclei_seg.py -c config_files/config_nuclei_seg_asdgan_server2.json -d 1 -r /data/zhennan/code_seg_cvpr/experiments/nuclei_seg/asdgan_unet/models/path_seg_fedgan/0827_124614/model_best.pth

python test_nuclei_seg.py -c config_files/config_nuclei_seg_asdgan_server1_x5.json -d 1 -r /data/zhennan/pytorch-template_aug11/experiments/nuclei_seg/asdgan_unet/models/path_seg_asdgan/1001_165122/model_best.pth

# real all
#python test_nuclei_seg.py -c config_files/config_nuclei_seg_real_server2.json -d 3 -r /data/zhennan/code_seg_cvpr/experiments/nuclei_seg/fedgan_unet/models/path_seg_real/0827_221136/checkpoint-epoch88.pth


# real breast, better
#python test_nuclei_seg.py -c config_files/config_nuclei_seg_real_breast_server2.json -d 1 -r /data/zhennan/code_seg_cvpr/experiments/nuclei_seg/unet/models/path_seg_real/0827_223109/model_best.pth



## fedseg
#python test_nuclei_seg.py -c config_files/config_nuclei_seg_fedseg_server2.json -d 1 -r /data/zhennan/FedML/fedml_experiments/distributed/fedseg/run/path/unet/experiment_4/aggregated_checkpoint_100.pth.tar
