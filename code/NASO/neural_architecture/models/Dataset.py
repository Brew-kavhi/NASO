import os
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from django.db import models
from django.http import JsonResponse

from helper_scripts.importing import get_class
from plugins.interfaces.Dataset import DatasetLoaderInterface


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
        super(DatasetLoader, self).save(*args, **kwargs)

    def __str__(self):
        return self.name

    def load_dataset_loader(self):
        return get_class(self.module_name, self.class_name)()

    def get_data(self, *args, **kwargs):
        if not self.dataset_loader:
            self.dataset_loader = self.load_dataset_loader()
        return self.dataset_loader.get_data(*args, **kwargs)

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
        default=os.path.join(BASE_DIR, "datasets"), max_length=100, null=True
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
    remote_source = models.CharField(max_length=255, null=True)


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


def get_datasets(request, pk):
    dataset_loader = DatasetLoader.objects.get(pk=pk)
    datasets = dataset_loader.get_datasets()
    return JsonResponse(datasets, safe=False)
