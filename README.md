# Off-the-shelf learning: A way to use deep learning on MRI studies with a small sample size


### Create autoencoder
```
python run_autoencoder.py tf_records 32,34,128,256 2 
```

### Transfer learning from autoencoder
```
python run_transfer_learning_from_autoencoder.py AGE GM OFF_THE_SHELF resources/autoencoder.h5 resources/train_val_test.csv resources/external_test.csv 19 2 --num_epochs 2
```
