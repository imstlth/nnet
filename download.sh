wget -q https://github.com/imstlth/nnet/releases/download/v0.2.1/nnet https://github.com/imstlth/nnet/raw/main/train-images.idx3-ubyte https://github.com/imstlth/nnet/raw/main/train-labels.idx1-ubyte https://github.com/imstlth/nnet/raw/main/encoded.nnet
chmod u+x nnet
echo pour lancer le programme, écrivez ./nnet -D \$PWD/ 784,183,42,10
