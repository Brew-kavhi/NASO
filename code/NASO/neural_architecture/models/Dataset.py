import os
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from django.db import models


# TODO overthink this, as e should be able to first of all, curl some data, \
# and second of all to use builtin functions that load data.
class Dataset(models.Model):
    BASE_DIR = Path(__file__).resolve().parent.parent

    name = models.CharField(max_length=255, unique=True)
    description = models.TextField()
    upload_date = models.DateTimeField(auto_now_add=True)
    data_dir = models.CharField(
        default=os.path.join(BASE_DIR, "datasets"), max_length=100, null=True
    )
    as_supervised = models.BooleanField(default=True)

    def __str__(self):
        return self.name

    def get_data(self):
        (train_dataset, test_dataset) = tfds.load(
            self.name,
            split=["train", "test"],
            shuffle_files=True,
            data_dir=self.data_dir,
            as_supervised=self.as_supervised,
        )
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        return (train_dataset, test_dataset)
