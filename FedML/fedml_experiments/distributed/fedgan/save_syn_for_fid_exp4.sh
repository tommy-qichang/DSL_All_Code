for i in {10..200..10}
do
  python save_syn.py --cfg exp_4.yml --batch_size 1 --save_dir ./run/brats/dadgan_mc/experiment_0 --epoch ${i} --GPUid 0 --save_data
  mv ./results/dadgan_mc/test_${i}/brats_resnet_9blocks_epoch${i}_experiment_0.h5 /data/datasets/asdgan_data/fedgan_syn/brats_h5_all/
done
