
for i in {10..200..10}
do
  python save_syn.py --cfg n_exp_4_miss_mod.yml --batch_size 1 --save_dir ./run/brats/asdgan_mc_mm/experiment_0 --epoch ${i} --GPUid 3 --save_data
  mv ./results/asdgan_mc_mm/test_${i}/brats_resnet_9blocks_epoch${i}_experiment_0.h5 ../../../../datasets/asdgan_syn/brats_h5_all_mm/
done

