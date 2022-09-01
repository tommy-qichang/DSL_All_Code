echo "Downloading Cardiac dataset into DSL and FedML"

TARGET_DIR=./datasets/cardiac
mkdir -p $TARGET_DIR
wget -N https://people.cs.rutgers.edu/~qc58/FedDGAN/cardiac/cardiac.zip -O $TARGET_DIR/cardiac.zip --no-check-certificate
tar -zxvf $TARGET_DIR/cardiac.zip -C $TARGET_DIR
rm $TARGET_DIR/cardiac.zip

echo "Finish downloading Cardiac datasets. Please change the configurations in each experiment."
