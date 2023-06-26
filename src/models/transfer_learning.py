import gc
import logging

import keras
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from keras import Sequential
from keras.callbacks import History

from src.base.data_manager import TrainingDataManager
from src.base.data_types import BrainTissue, TrainingStrategy
from src.models.model_checkpoint import ModelCheckpointWarmUp
from src.utils import load_data_from_df


class TransferLearningResult:
    def __init__(self, model: Sequential, fit_history: History, train_scores: dict,
                 val_scores: dict, test_scores: dict, external_test_scores: dict):
        self.model = model
        self.fit_history = fit_history

        self.train_scores = train_scores
        self.val_scores = val_scores
        self.test_scores = test_scores
        self.external_test_scores = external_test_scores


class TransferLearningAutoencoder:
    LOSS = None
    METRICS_EVALUATE = None
    ACTIVATION_LAST_LAYER = None
    EARLY_STOPPING_MODE = "auto"
    MIN_DELTA = 0

    def __init__(self, path_autoencoder: str, layer_encoding_stop: int,
                 training_strategy: TrainingStrategy, brain_tissue: BrainTissue):

        self.layer_encoding_stop = layer_encoding_stop
        self.training_strategy = training_strategy
        self.brain_tissue = brain_tissue
        self.encoding_model = self.load_autoencoder_model(path_autoencoder)

        self.model_layers = [
            layers.Conv3D(64, (2, 2, 2), strides=1, name="regression-block1_conv", activation=None,
                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=2022)),
            layers.BatchNormalization(name="regression-block1_batch"),
            layers.MaxPooling3D(pool_size=(2, 2, 2), name="regression-block1_pool"),
            layers.ReLU(name="regression-block1_ReLU"),

            layers.Conv3D(32, (2, 2, 2), strides=1, name="regression-block2_conv", activation=None,
                          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=2022)),
            layers.BatchNormalization(name="regression-block2_batch"),
            layers.MaxPooling3D(pool_size=(2, 2, 2), name="regression-block2_pool"),
            layers.ReLU(name="regression-block2_ReLU"),

            layers.Flatten(name="regression-block3_flatten"),
            layers.Dropout(0.25),
            layers.Dense(1, name="regression-block3_dense1", activation=self.ACTIVATION_LAST_LAYER,
                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=2022))

        ]

        if self.METRICS_EVALUATE is None:
            raise AttributeError("Please define METRIC_EVALUATE on your class")
        if self.LOSS is None:
            raise AttributeError("Please define LOSS on your class")

    def load_autoencoder_model(self, path_autoencoder: str):

        model_original = keras.models.load_model(path_autoencoder)

        if self.training_strategy == TrainingStrategy.TRAINING_FROM_SCRATCH:
            model_original = tf.keras.models.clone_model(
                model_original, input_tensors=None, clone_function=None
            )

        encoding_model = tf.keras.Sequential(model_original.layers[:self.layer_encoding_stop])

        return encoding_model

    def create_model(self):

        final_model = keras.Sequential()
        final_model.add(layers.Input(shape=(121, 145, 121, 1)))

        for i, model_layer in enumerate(self.encoding_model.layers):
            final_model.add(model_layer)
            if i > 1 and self.training_strategy == TrainingStrategy.OFF_THE_SHELF:
                final_model.layers[i].trainable = False

        for personalized_layers in self.model_layers:
            final_model.add(personalized_layers)

        logging.info(final_model.summary())

        final_model.compile(optimizer='adam', loss=self.LOSS, metrics=self.METRICS_EVALUATE)

        return final_model

    def split_into_datasets(self, df_processed: pd.DataFrame, column_abs_path: str, column_label: str,
                            number_training_samples: int, number_validation_samples: int, number_test_samples: int):
        return TrainingDataManager().stratified_spliter(df_processed, column_abs_path, column_label,
                                                        number_training_samples, number_validation_samples,
                                                        number_test_samples)

    def get_data_train_val_and_test(self, df_processed: pd.DataFrame, column_abs_path: str, column_label: str,
                                    number_training_samples: int, number_validation_samples: int,
                                    number_test_samples: int):
        df_train, df_val, df_test = self.split_into_datasets(df_processed, column_abs_path, column_label,
                                                             number_training_samples, number_validation_samples,
                                                             number_test_samples)
        return load_data_from_df(df_train, column_abs_path, column_label), \
            load_data_from_df(df_val, column_abs_path, column_label), \
            load_data_from_df(df_test, column_abs_path, column_label)

    def train_all_data(self, model, batch_size, X_train, y_train, X_val, y_val, epochs: int, filename_model):

        model_checkpoint = ModelCheckpointWarmUp(filename_model, max(0, epochs - 30), save_freq='epoch',
                                                 save_best_only=True,
                                                 save_weights_only=False)

        return model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                         callbacks=[model_checkpoint])

    def train_evaluate_model(self, df_train_val_test: pd.DataFrame, df_test_external: pd.DataFrame, column_path: str,
                             column_label: str, num_train_samples: int, num_val_samples: int, num_test_samples: int,
                             batch_size: int, num_epochs: int, filename_best_model: str):

        model_ = self.create_model()

        train_data, val_data, test_data = self.get_data_train_val_and_test(df_train_val_test,
                                                                           column_path, column_label,
                                                                           num_train_samples,
                                                                           num_val_samples,
                                                                           num_test_samples)
        external_test_data = load_data_from_df(df_test_external, column_path, column_label)

        logging.info("Loaded complete")
        fit_history = self.train_all_data(model_, batch_size, train_data[0], train_data[1],
                                          val_data[0], val_data[1], num_epochs, filename_best_model)

        best_model = keras.models.load_model(filename_best_model)

        gc.collect()
        train_score = best_model.evaluate(train_data[0], train_data[1], batch_size, return_dict=True)
        val_score = best_model.evaluate(val_data[0], val_data[1], batch_size, return_dict=True)
        test_score = best_model.evaluate(test_data[0], test_data[1], batch_size, return_dict=True)
        external_test_score = best_model.evaluate(external_test_data[0], external_test_data[1], batch_size,
                                                  return_dict=True)

        logging.info("Loading all data")

        return TransferLearningResult(best_model, fit_history, train_score, val_score, test_score,
                                      external_test_score)


class RegressionTransferLearningAutoencoder(TransferLearningAutoencoder):
    LOSS = "mse"
    METRICS_EVALUATE = [tf.keras.metrics.MeanAbsoluteError()]

    ACTIVATION_LAST_LAYER = None

    def split_into_datasets(self, df_processed: pd.DataFrame, column_abs_path: str, column_label: str,
                            number_training_samples: int, number_validation_samples: int, number_test_samples: int):
        df = TrainingDataManager().binarization(df_processed, column_label, "bins", 15)
        return TrainingDataManager().stratified_spliter(df, column_abs_path, "bins",
                                                        number_training_samples, number_validation_samples,
                                                        number_test_samples)


class ClassificationTransferLearningAutoencoder(TransferLearningAutoencoder):
    LOSS = "binary_crossentropy"
    METRICS_EVALUATE = [tf.keras.metrics.BinaryAccuracy()]

    ACTIVATION_LAST_LAYER = "sigmoid"
