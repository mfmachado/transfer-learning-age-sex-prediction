import argparse
import datetime
import logging

import tensorflow as tf


from src.models.autoencoder import AutoEncoder
from src.base.tf_recod_manager import MRITFRecordLoader


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("path_tf_records", help="Directory with tf records", type=str)
    parser.add_argument("architecture", help="# of layers - separate the layers number by comma (,)", type=str)
    parser.add_argument("batch_size", help="batch size", type=int)
    parser.add_argument('-e', '--number_of_epochs', help="Number of epochs to train the model", type=int,
                        default=150)
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    arg = parse_arguments()
    logging.basicConfig(level=logging.INFO)

    path_tf_records = arg.path_tf_records

    number_layers = [int(el) for el in arg.architecture.split(",")]
    batch_size = arg.batch_size
    number_of_epochs = arg.number_of_epochs

    autoencoder = AutoEncoder().create(number_layers)
    logging.info(autoencoder.summary())

    logging.info(f"Batchsize: {batch_size}")

    logging.info("Loading training dataset")
    mri_record_loader = MRITFRecordLoader(path_tf_records, class_mode="input", batch_size=batch_size)
    train_dataset = mri_record_loader.get_dataset("train*")
    logging.info("Loading validation dataset")
    valid_dataset = mri_record_loader.get_dataset("val*")

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='10,150')

    result = autoencoder.fit(train_dataset,
                             epochs=number_of_epochs,
                             validation_data=valid_dataset,
                             callbacks=[tensorboard_callback]
                             )

    autoencoder.save("resources/autoencoder.h5")
