![Python-Version](https://img.shields.io/badge/python-3.10-blue)

# Overcoming Data Availability in MRI Studies: Leveraging Off-the-Shelf Deep Learning Models


![encoder](https://github.com/mfmachado/transfer-learning-age-sex-prediction/blob/main/resources/autoencoder.png)


### Create autoencoder
```
path_tf_recods: Path to TF Records
encoder_architecture: Number of filter in each convolution layers of the encoder block, the decoder mirrors the encoder. The number are separated by comma
batch_size: Batch size

python run_autoencoder.py $path_tf_recods $encoder_architecture $batch_size
```

### Transfer learning from autoencoder
```
problem={AGE,GENDER}
tissue={GM,WM,CSF,DF}
training_strategy={OFF_THE_SHELF,FINE_TUNING,TRAINING_FROM_SCRATCH}
path_autoencoder: Path autoencoder file, can use the one on resources (resources/autoencoder.h5)
path_training_val_test_data: Path to csv file with data to use as train, validation and test (Exemple in:resources/train_val_test.csv)
path_external_test_data: Path to csv file with data to use as external test (Exemple in:resources/external_test.csv)
cutoff_layer: Number of autoencoder layers to consider in the model
batch_size: Batch size
number_epochs: Bumber of epochs to train the model
num_train_samples: Number of instances to train the model
num_val_samples: Number of instances to validate the model
num_test_samples: Number of instances to test the model

python run_transfer_learning_from_autoencoder.py $problem $tissue $training_strategy $path_autoencoder $path_training_val_test_data $path_external_test_data $cutoff_layer $batch_size --num_epochs $number_epochs --num_train_samples $num_train_samples --num_val_samples $num_val_samples --num_test_samples $num_test_samples
```
