import tensorflow as tf
from django.db import models


# TODO overthink this, as e should be able to first of all, curl some data, \
# and second of all to use builtin functions that load data.
class Dataset(models.Model):
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField()
    upload_date = models.DateTimeField(auto_now_add=True)
    data_file = models.FileField(upload_to="datasets/")

    def __str__(self):
        return self.name


def load_keras_mist_dataset():
    return tf.keras.datasets.mnist.load_data()
