Data can be downloaded from [here](https://drive.google.com/drive/folders/1dTSRzUdBRlz8NFcANnrbuz-XicnYa97_?usp=sharing).

To run code well, it's necessary to download the following directories and put them in this directory without changing thier names.

1. `XNLI-MT-1.0`
2. `XNLI-15way`
3. `XNLI-1.0`


Other files used to fine-tune models are as follows:

e.g. `cached_train_xnli_en_3e-5_2.0_128_xnli_de` file is used for the model that is fine-tuned with English first and uses this cache file to continue fine-tuning on German.

Parameters for that chshe file are:
```
learning rate = 3e-5
epochs = 2.0
max_length = 128
task = xnli
```
