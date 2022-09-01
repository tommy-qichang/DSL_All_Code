echo "Downloading Nuclei dataset into DSL and FedML"

TARGET_DIR=./datasets/hist
mkdir -p $TARGET_DIR
wget -N https://people.cs.rutgers.edu/~qc58/FedDGAN/Hist/hist.zip -O $TARGET_DIR/hist.zip --no-check-certificate
tar -zxvf $TARGET_DIR/hist.zip -C $TARGET_DIR
rm $TARGET_DIR/hist.zip

echo "Finish downloading Nuclei datasets. Please change the configurations in each experiment."
