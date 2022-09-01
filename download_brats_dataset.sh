echo "Downloading BraTS dataset for DSL and FedML"

TARGET_DIR=./datasets/brats
mkdir -p $TARGET_DIR
wget -N https://people.cs.rutgers.edu/~qc58/FedDGAN/brats/brats.zip -O $TARGET_DIR/brats.zip --no-check-certificate
tar -zxvf $TARGET_DIR/brats.zip -C $TARGET_DIR
rm $TARGET_DIR/brats.zip

echo "Finish downloading BraTS datasets. Please change the configurations in each experiment."
