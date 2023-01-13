for i in {10..200..10}
do
  python save_syn.py --cfg exp_1.yml --batch_size 1 --save_dir ./run/heart/dadgan/experiment_0 --epoch ${i} --GPUid 3 --save_data
  mv ./results/dadgan/test_${i}/heart_resnet_9blocks_epoch${i}_experiment_0.h5  ../../../../datasets/fedgan_syn/heart_h5_all/
done
