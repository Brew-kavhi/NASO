import os
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from django.db import models
from django.http import JsonResponse
from sklearn import datasets

from helper_scripts.importing import get_class
from plugins.interfaces.dataset import DatasetLoaderInterface


class DatasetLoader(models.Model):
    module_name = models.CharField(max_length=255)
    class_name = models.CharField(max_length=64)
    name = models.CharField(max_length=64)
    description = models.TextField()

    dataset_loader = None

    class Meta:
        unique_together = ["class_name", "module_name"]

    def save(self, *args, **kwargs):
        self.dataset_loader = self.load_dataset_loader()
        if not issubclass(type(self.dataset_loader), DatasetLoaderInterface):
            raise TypeError(
                f"DatasetLoader {self} is not a subclass of tfds.core.DatasetBuilder"
            )
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

    def load_dataset_loader(self):
        return get_class(self.module_name, self.class_name)()

    def get_data(self, *args, **kwargs) -> (tf.data.Dataset, tf.data.Dataset):
        """
        Returns a tuple of two tf.data.Dataset objects, the first one is the training dataset,
        the second one is the test dataset
        """
        if not self.dataset_loader:
            self.dataset_loader = self.load_dataset_loader()
        data = self.dataset_loader.get_data(*args, **kwargs)
        train_dataset = data[0]
        if len(data) > 1:
            test_dataset = data[1]
        else:
            test_dataset = train_dataset
        return (train_dataset, test_dataset)

    def get_datasets(self):
        if not self.dataset_loader:
            self.dataset_loader = self.load_dataset_loader()
        return self.dataset_loader.get_datasets()


class Dataset(models.Model):
    BASE_DIR = Path(__file__).resolve().parent.parent

    name = models.CharField(max_length=255)
    description = models.TextField()
    upload_date = models.DateTimeField(auto_now_add=True)
    data_dir = models.CharField(
        default=os.path.join(BASE_DIR, "datasets"), max_length=100
    )
    as_supervised = models.BooleanField(default=True)
    dataset_loader = models.ForeignKey(
        DatasetLoader, on_delete=models.CASCADE, null=True
    )

    class Meta:
        unique_together = ["name", "as_supervised", "dataset_loader"]

    def __str__(self):
        return self.name

    def get_data(self, *args, **kwargs):
        return self.dataset_loader.get_data(
            name=self.name,
            as_supervised=self.as_supervised,
            data_dir=self.data_dir,
            *args,
            **kwargs,
        )


class LocalDataset(models.Model):
    file = models.FileField(upload_to="datasets/")
    name = models.CharField(max_length=64)
    dataloader = models.ForeignKey(DatasetLoader, on_delete=models.CASCADE, null=True)
    remote_source = models.CharField(max_length=255)


class TensorflowDatasetLoader(DatasetLoaderInterface):
    module_name = "tensorflow_datasets"
    dataset_list = tfds.list_builders()

    def get_data(self, name, as_supervised, *args, **kwargs):
        (train_dataset, test_dataset) = tfds.load(
            name,
            split=["train", "test"],
            as_supervised=as_supervised,
        )
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        return (train_dataset, test_dataset)

    def get_datasets(self, *args, **kwargs):
        return self.dataset_list


class SkLearnDatasetLoader(DatasetLoaderInterface):
    dataset_list = {
        "California Housing": "fetch_california_housing",
        "RCV1": "fetch_rcv1",
        "20 Newsgroup": "fetch_20newsgroups",
        "Olivetti Faces": "fetch_olivetti_faces",
        "Forest Covertypes": "fetch_covtype",
        "LFW People": "fetch_lfw_people",
        "Kid cup": "fetch_kddcup99",
        "Iris plants (clasification)": "load_iris",
        "Diabetes (regression)": "load_diabetes",
        "Handwritten digits (calssifation)": "load_digits",
        "Physical excercise Linnerud (regression)": "load_linnerud",
        "Wines": "load_wine",
        "Breast cancer": "load_breast_cancer",
    }
    training_split = 0.9

    def get_datasets(self, *args, **kwargs):
        return list(self.dataset_list.keys())

    def get_data(self, name, *args, **kwargs):
        loader_args = {}
        if "data_dir" in args:
            loader_args["data_home"] = args["data_dir"]

        data = getattr(datasets, self.dataset_list[name])(**loader_args)
        target_values = data.target.reshape((-1, 1))
        size = target_values.shape[0]

        train_set = tf.data.Dataset.from_tensor_slices(
            (
                data.data[: int(size * self.training_split)],
                target_values[: int(size * self.training_split)],
            )
        )
        test_set = tf.data.Dataset.from_tensor_slices(
            (
                data.data[int(size * self.training_split) :],
                target_values[int(size * self.training_split) :],
            )
        )
        return (train_set, test_set)


def get_datasets(request, pk):
    dataset_loader = DatasetLoader.objects.get(pk=pk)
    datasets = dataset_loader.get_datasets()
    return JsonResponse(datasets, safe=False)
