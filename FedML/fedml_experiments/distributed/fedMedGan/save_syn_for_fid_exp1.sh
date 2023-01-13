for i in {10..100..10}
do
  python save_syn.py --cfg exp_1.yml --batch_size 1 --save_dir ./run/heart/fedmedgan/experiment_0 --epoch ${i} --GPUid 3 --save_data
  mv ./results/fedmedgan/test_${i}/heart_resnet_9blocks_epoch${i}_experiment_0.h5  ../../../../datasets/fedmedgan_syn/heart_h5_all/
done
