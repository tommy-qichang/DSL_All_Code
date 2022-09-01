
python test_general_brats.py -c config_files/config_BRATS_seg_exp4.json -r saved/models/brats_seg_AsynDGANv2_exp4/1012_162127/checkpoint-epoch47.pth

python test_general_brats.py -c config_files/config_BRATS_seg_exp6.json -r saved/models/brats_seg_AsynDGANv2_exp6/1012_162214/checkpoint-epoch33.pth

python test_general_brats.py -c config_files/config_BRATS_seg_exp8.json -r  saved/models/brats_seg_AsynDGANv2_exp8/1012_162306/checkpoint-epoch21.pth


python test_general_brats.py -c config_files/modalitybank_rebuttal_lgg_pretrain.json -r  saved/models/modalitybank_rebuttal_lgg_t1t2flair_epoch35/0518_175224/checkpoint-epoch12.pth

python test_general_brats.py -c config_files/modalitybank_rebuttal_mms_pretrain.json -r  saved/models/modalitybank_rebuttal_mms_t1t2flair_epoch35/0518_175202/checkpoint-epoch11.pth


python test_general_brats.py -c config_files/config_fsl_seg_exp4-6.json -r saved/models/fsl_brats_seg_exp4-6/0823_171714/checkpoint-epoch70.pth

python test_general_brats.py -c config_files/config_fsl_seg_exp4-6_da.json -r saved/models/fsl_brats_seg_exp4-6_da/1002_011754/checkpoint-epoch11.pth




