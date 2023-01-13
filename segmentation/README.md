# FedD-GAN Segmentation Code


## Folder Structure

```
  segmentation/
  │
  ├── config_files - include experiment's configuration files
  ├── data_loader - general data loader and dataset with my_transforms
  │── loss - including MSE, DICE loss etc.
  │── metrics - evaluate metrics
  │── model - UNet2D architecture
  │── loss - including MSE, DICE loss etc.
  │── train.py - main entrance for segmentation task.
```

## Command for training the segmentation

Please change the path of data_loader->data_dir in config_files/*.json

```
    export PATH=.;$PATH
    cd segmentation
    #Train experiments on real-all data
    python train.py -c config_files/config_fsl_seg_exp1-1.json -d GPU_ID
    #Train experiments on WHS data
    python train.py -c config_files/config_fsl_seg_exp1-3.json -d GPU_ID
    #Train experiments on CAT08 data
    python train.py -c config_files/config_fsl_seg_exp1-4.json -d GPU_ID
    #Train experiments on ASOCA data
    python train.py -c config_files/config_fsl_seg_exp1-5.json -d GPU_ID
    #Train experiments on synthetic FedD-GAN data
    python train.py -c config_files/config_fsl_seg_exp1-6-3.json -d GPU_ID
    
    
    
    
    
    
```














