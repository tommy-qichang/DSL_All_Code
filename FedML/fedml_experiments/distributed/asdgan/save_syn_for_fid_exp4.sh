
for i in {10..200..10}
do
  python save_syn.py --cfg n_exp_4.yml --batch_size 1 --save_dir ./run/brats/asdgan/experiment_0 --epoch ${i} --GPUid 2 --save_data
  mv ./results/asdgan/test_${i}/brats_resnet_9blocks_epoch${i}_experiment_0.h5 /data/datasets/asdgan_data/asdgan_syn/brats_h5_all/
done
