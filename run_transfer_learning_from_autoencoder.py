import argparse
import logging
import random
import tensorflow as tf
import numpy as np
import pandas as pd

from src.base.data_types import BrainTissue, DATASETS, TrainingStrategy, ProblemOptimize
from src.models.transfer_learning import RegressionTransferLearningAutoencoder, \
    ClassificationTransferLearningAutoencoder


def parse_arguments():
    prob_choices = [ProblemOptimize.AGE.name, ProblemOptimize.GENDER.name]
    training_choices = [el.name for el in TrainingStrategy]
    tissue_choices = [el.name for el in BrainTissue]

    parser = argparse.ArgumentParser()
    parser.add_argument("problem", choices=prob_choices, help="Problem to optimize", type=str)
    parser.add_argument("tissue", choices=tissue_choices, help="Tissue type to train", type=str)
    parser.add_argument("training_strategy", choices=training_choices, help="Training strategy", type=str)
    parser.add_argument("autoencoder_path", help="Autoencoder path", type=str)
    parser.add_argument("data_train_val_test_path", help="CSV file with data to train, validate and test", type=str)
    parser.add_argument("data_external_test_path", help="CSV file with external test data", type=str)
    parser.add_argument("cutoff_layer_encoder", help="Cutoff layer", type=int)
    parser.add_argument("batch_size", help="Batch size", type=int)
    parser.add_argument("-t", "--num_train_samples", help="Number of training samples", type=int, default=100)
    parser.add_argument("-v", "--num_val_samples", help="Number of validation samples", type=int, default=92)
    parser.add_argument("-m", "--num_test_samples", help="Number of test samples", type=int, default=100)
    parser.add_argument("-e", "--num_epochs", help="Number of training epochs", type=int, default=150)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    logging.basicConfig(level=logging.INFO)

    tf.random.set_seed(seed)
    # tf.keras.utils.set_random_seed(seed)
    # tf.config.experimental.enable_op_determinism()

    args = parse_arguments()

    problem_optimize = ProblemOptimize[args.problem]
    tissue = BrainTissue[args.tissue]

    training_strategy = TrainingStrategy[args.training_strategy]

    autoencoder_path = args.autoencoder_path
    path_train_val_test = args.data_train_val_test_path
    path_external_test = args.data_external_test_path

    cutoff_layer_encoder = args.cutoff_layer_encoder
    batch_size = args.batch_size

    num_train_samples = args.num_train_samples
    num_val_samples = args.num_val_samples
    num_test_samples = args.num_test_samples
    num_epochs = args.num_epochs

    repository = DATASETS.IXI

    column_path = "path_abs"

    # Load Subjects info

    df_train_val_test = pd.read_csv(path_train_val_test)
    df_test_external = pd.read_csv(path_external_test)

    logging.info(f"Start training with a batch size of {batch_size}")

    filename_best_model = f"model_checkpoints/brain-age-{num_train_samples}-{training_strategy}" \
                          f"-{cutoff_layer_encoder}-{batch_size}-{problem_optimize}.h5"

    df_train_val_test = df_train_val_test.loc[~df_train_val_test[problem_optimize.name.lower()].isna()]
    df_test_external = df_test_external.loc[~df_test_external[problem_optimize.name.lower()].isna()]

    for it in range(30):

        df_train_val_test = df_train_val_test.sort_values([column_path]).sample(frac=1)

        if problem_optimize == ProblemOptimize.AGE:
            transfer_autoencoder = RegressionTransferLearningAutoencoder(autoencoder_path, cutoff_layer_encoder,
                                                                         training_strategy, tissue)
        elif problem_optimize == ProblemOptimize.GENDER:
            transfer_autoencoder = ClassificationTransferLearningAutoencoder(autoencoder_path, cutoff_layer_encoder,
                                                                             training_strategy, tissue)
        else:
            raise NotImplementedError

        result = transfer_autoencoder.train_evaluate_model(df_train_val_test, df_test_external, column_path,
                                                           problem_optimize.name.lower(),
                                                           num_train_samples, num_val_samples, num_test_samples,
                                                           batch_size, num_epochs, filename_best_model)

        logging.info(f"Scores | train: {result.train_scores}; val: {result.val_scores}; test: {result.test_scores}: "
                     f"external test: {result.external_test_scores}")
        logging.info("Train done, start another iteration")
