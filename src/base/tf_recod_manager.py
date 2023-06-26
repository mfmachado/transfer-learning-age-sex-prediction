import logging
import os.path
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import List

import nibabel as nib
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordWriter


class MRITFRecord:
    CLASS_MODE_VALUES = ["input", "raw"]

    def __init__(self, path_records):
        self.path_records = path_records

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))


class MRITFRecordCreator(MRITFRecord):

    def __init__(self, path_records, n_jobs=1, number_mri_per_tf_record: int = 4):
        super().__init__(path_records)
        self.n_jobs = n_jobs
        self.number_mri_per_tf_record = number_mri_per_tf_record

    def load_img(self, path_img):
        if not os.path.exists(path_img):
            err_msg = f"The file {path_img} does not exist"
            logging.error(err_msg)
            raise FileNotFoundError(err_msg)
        return nib.load(path_img).get_fdata()

    def create_data_records(self, out_filename: str, img_paths: List[str], img_info: List[dict] = None,
                            clean_dir: bool = True):

        # open the TFRecords file
        if clean_dir:
            logging.info("Cleaning directory")
            if os.path.exists(self.path_records):
                shutil.rmtree(self.path_records)

        Path(self.path_records).mkdir(parents=True, exist_ok=True)

        logging.info(f"Creating dataset with {len(img_paths)} instances")

        if img_info is None:
            chunks = [(os.path.join(self.path_records, f"{out_filename}_{k}.tfrecords"),
                       img_paths[x:x + self.number_mri_per_tf_record])
                      for k, x in enumerate(range(0, len(img_paths), self.number_mri_per_tf_record))]
        else:
            chunks = [(os.path.join(self.path_records, f"{out_filename}_{k}.tfrecords"),
                       img_paths[x:x + self.number_mri_per_tf_record], img_info[x:x + self.number_mri_per_tf_record])
                      for k, x in enumerate(range(0, len(img_paths), self.number_mri_per_tf_record))]

        logging.info(f"The data will be divided into {len(chunks)} records")
        logging.info(f"Starting a Pool with {self.n_jobs} processes")

        with Pool(processes=self.n_jobs) as pool:
            pool.starmap(self.create_data_record, chunks)

        logging.info("Creating dataset: done")

    def create_data_record(self, file_name_record: str, img_paths: List[str], img_info: List[dict] = None):

        writer = TFRecordWriter(file_name_record)

        for i in range(len(img_paths)):

            # Load the image
            img = self.load_img(img_paths[i])

            if img is None:
                continue

            # Create a feature
            feature = {
                'image': self._bytes_feature(img.ravel()),
            }
            if img_info is not None:
                feature.update({label: self._float_feature(img_info[i][label]) for label in img_info[i]})

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        writer.close()


class MRITFRecordLoader(MRITFRecord):
    AUTOTUNE = tf.data.AUTOTUNE
    IMAGE_SIZE = (121, 145, 121)

    def __init__(self, path_records, class_mode: str, batch_size: int, label_name: str = None):
        super().__init__(path_records)

        self.class_mode = class_mode
        self.batch_size = batch_size
        self.label_name = label_name

        if class_mode not in self.CLASS_MODE_VALUES:
            err_msg = f"The class_mode attribute value should have one of the following values:" \
                      f" {self.CLASS_MODE_VALUES}"
            logging.error(err_msg)
            raise AttributeError(err_msg)
        if class_mode == "input" and label_name is not None:
            err_msg = f"If the class_mode value is {class_mode}, then the label should be " \
                      f"None, because in this case the output is the image itself."
            logging.error(err_msg)
            raise AttributeError(err_msg)

        if class_mode != "input" and label_name is None:
            err_msg = f"If the class_mode value is {class_mode}, then the label should be defined."
            logging.error(err_msg)
            raise AttributeError(err_msg)

    def decode_image(self, image):
        image = tf.io.decode_raw(image, out_type=tf.float64)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, self.IMAGE_SIZE)
        return image

    def load_dataset(self, filename_pattern):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed

        list_ds = tf.data.Dataset.list_files(os.path.join(self.path_records, filename_pattern))

        dataset = list_ds.interleave(lambda x: tf.data.TFRecordDataset(x).map(self.read_tfrecord,
                                                                              num_parallel_calls=tf.data.AUTOTUNE),
                                     num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def read_tfrecord(self, example):

        tfrecord_format = (
            {
                "image": tf.io.FixedLenFeature([], tf.string),
                self.label_name: tf.io.FixedLenFeature([], tf.float32),
            }
            if self.class_mode != "input"
            else {"image": tf.io.FixedLenFeature([], tf.string), }
        )
        example = tf.io.parse_single_example(example, tfrecord_format)
        image = self.decode_image(example["image"])
        if self.class_mode != "input":
            return image, example[self.label_name]
        return image, image

    def get_dataset(self, filename_pattern):
        dataset = self.load_dataset(filename_pattern)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset = dataset.with_options(options)
        return dataset

