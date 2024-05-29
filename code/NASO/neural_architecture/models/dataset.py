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
    """
    Represents a dataset loader that loads and provides access to training and test datasets.

    Attributes:
        module_name (str): The name of the module containing the dataset loader class.
        class_name (str): The name of the dataset loader class.
        name (str): The name of the dataset loader.
        description (str): The description of the dataset loader.
        dataset_loader (DatasetLoaderInterface): The instance of the dataset loader class.

    Methods:
        save(*args, **kwargs): Overrides the save method to validate the dataset loader.
        __str__(): Returns the name of the dataset loader.
        load_dataset_loader(): Loads the dataset loader class.
        get_element_size(*args, **kwargs): Returns the size of an element in the dataset.
        get_data(*args, **kwargs): Returns the training and test datasets.
        get_datasets(): Returns the datasets provided by the dataset loader.
    """

    module_name = models.CharField(max_length=255)
    class_name = models.CharField(max_length=64)
    name = models.CharField(max_length=64)
    description = models.TextField()

    dataset_loader = None

    class Meta:
        unique_together = ["class_name", "module_name"]

    def save(self, *args, **kwargs):
        """
        Overrides the save method to validate the dataset loader.

        Raises:
            TypeError: If the dataset loader is not a subclass of tfds.core.DatasetBuilder.
        """
        self.dataset_loader = self.load_dataset_loader()
        if not issubclass(type(self.dataset_loader), DatasetLoaderInterface):
            raise TypeError(
                f"DatasetLoader {self} is not a subclass of tfds.core.DatasetBuilder"
            )
        super().save(*args, **kwargs)

    def __str__(self):
        """
        Returns the name of the dataset loader.
        """
        return self.name

    def load_dataset_loader(self):
        """
        Loads the dataset loader class.

        Returns:
            DatasetLoaderInterface: An instance of the dataset loader class.
        """
        return get_class(self.module_name, self.class_name)()

    def get_element_size(self, *args, **kwargs):
        """
        Returns the size of an element in the dataset.

        Raises:
            Exception: If no dataset is loaded yet.

        Returns:
            int: The size of an element in the dataset.
        """
        if not self.dataset_loader:
            raise ValueError("no dataset loaded yet")
        return self.dataset_loader.get_element_size()

    def get_data(
        self, *args, **kwargs
    ) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        """
        Returns the training and test datasets.

        Returns:
            tuple: A tuple of two tf.data.Dataset objects, the first one is the training dataset,
            and the second one is the test dataset.
        """
        if not self.dataset_loader:
            self.dataset_loader = self.load_dataset_loader()
        data = self.dataset_loader.get_data(*args, **kwargs)
        train_dataset = data[0]
        if len(data) > 1:
            test_dataset = data[1]
        else:
            test_dataset = train_dataset
        if len(data) > 2:
            eval_dataset = data[2]
        else:
            eval_dataset = train_dataset
        return (train_dataset, test_dataset, eval_dataset)

    def get_datasets(self):
        """
        Returns the datasets provided by the dataset loader.

        Returns:
            list: A list of tf.data.Dataset objects.
        """
        if not self.dataset_loader:
            self.dataset_loader = self.load_dataset_loader()
        return self.dataset_loader.get_datasets()


class Dataset(models.Model):
    """
    Represents a dataset used in the application.

    Attributes:
        name (str): The name of the dataset.
        description (str): A description of the dataset.
        upload_date (datetime): The date and time when the dataset was uploaded.
        data_dir (str): The directory where the dataset is stored.
        as_supervised (bool): Indicates whether the dataset is supervised or not.
        dataset_loader (DatasetLoader): The dataset loader associated with the dataset.

    Methods:
        __str__(): Returns a string representation of the dataset.
        get_element_size(*args, **kwargs): Returns the size of an element in the dataset.
        get_data(*args, **kwargs): Returns the data from the dataset.

    """

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

    def get_element_size(self, *args, **kwargs):
        if not self.dataset_loader:
            raise ValueError("no dataset loaded yet")
        return self.dataset_loader.get_element_size()

    def get_data(self, *args, **kwargs):
        return self.dataset_loader.get_data(
            name=self.name,
            as_supervised=self.as_supervised,
            data_dir=self.data_dir,
            *args,
            **kwargs,
        )


class LocalDataset(models.Model):
    """
    Represents a local dataset.

    Attributes:
        file (FileField): The uploaded file for the dataset.
        name (CharField): The name of the dataset.
        dataloader (ForeignKey): The foreign key to the DatasetLoader model.
        remote_source (CharField): The remote source of the dataset.
    """

    file = models.FileField(upload_to="datasets/")
    name = models.CharField(max_length=64)
    dataloader = models.ForeignKey(DatasetLoader, on_delete=models.CASCADE, null=True)
    remote_source = models.CharField(max_length=255)


class TensorflowDatasetLoader(DatasetLoaderInterface):
    """
    A class for loading datasets using TensorFlow Datasets.

    Attributes:
        module_name (str): The name of the TensorFlow Datasets module.
        dataset_list (list): A list of available datasets in TensorFlow Datasets.
        info (dict): Information about the loaded dataset.

    Methods:
        normalize_img: Normalizes images from `uint8` to `float32`.
        get_data: Loads and preprocesses the specified dataset.
        get_size: Returns the size information of the dataset.
        get_element_size: Returns the size information of each element in the dataset.
        get_datasets: Returns the list of available datasets.
    """

    module_name = "tensorflow_datasets"
    dataset_list = tfds.list_builders()
    info: dict = {}

    def normalize_img(self, image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.0, label

    def get_data(self, name, as_supervised, *args, **kwargs):
        """
        Loads and preprocesses the specified dataset.

        Args:
            name (str): The name of the dataset to load.
            as_supervised (bool): Whether to load the dataset in supervised mode.

        Returns:
            tuple: A tuple containing the train and test datasets.
        """
        (dataset, self.info) = tfds.load(
            name,
            split=["train", "test[:50%]", "test[50%:]"],
            as_supervised=as_supervised,
            with_info=True,
        )
        (train_dataset, test_dataset, eval_dataset) = dataset
        train_dataset = train_dataset.map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE
        )
        test_dataset = test_dataset.map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE
        )
        eval_dataset = eval_dataset.map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        return (train_dataset, test_dataset, eval_dataset)

    def get_size(self, *args, **kwargs):
        """
        Returns the size information of the dataset.

        Returns:
            tuple: The shape of the dataset splits.

        Raises:
            Exception: If the dataset does not support size information.
        """
        if "splits" in self.info.features:
            return self.info.features["splits"].shape
        raise NotImplementedError("This dataset does not support size information")

    def get_element_size(self, *args, **kwargs):
        """
        Returns the size information of each element in the dataset.

        Returns:
            tuple: The shape of the dataset elements.

        Raises:
            Exception: If the dataset does not support size information.
        """
        if "image" in self.info.features:
            return self.info.features["image"].shape
        raise NotImplementedError("This dataset does not support size information")

    def get_datasets(self, *args, **kwargs):
        """
        Returns the list of available datasets.

        Returns:
            list: A list of available datasets in TensorFlow Datasets.
        """
        return self.dataset_list


class SkLearnDatasetLoader(DatasetLoaderInterface):
    """
    A class for loading datasets using scikit-learn.

    Attributes:
        dataset_list (dict): A dictionary mapping dataset names to scikit-learn dataset functions.
        training_split (float): The fraction of data to be used for training.
        element_size (int): The size of each element in the dataset.
        dataset_size (int): The total size of the dataset.
    """

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
    training_split = 0.8
    element_size = 0
    dataset_size = 0

    def get_size(self, *args, **kwargs):
        """
        Get the size of the dataset.

        Returns:
            int: The size of the dataset.
        """
        return self.dataset_size

    def get_element_size(self, *args, **kwargs):
        """
        Get the size of each element in the dataset.

        Returns:
            int: The size of each element in the dataset.
        """
        return self.element_size

    def get_datasets(self, *args, **kwargs):
        """
        Get the list of available datasets.

        Returns:
            list: A list of dataset names.
        """
        return list(self.dataset_list.keys())

    def get_data(self, name, *args, **kwargs):
        """
        Get the train and test sets for a specific dataset.

        Args:
            name (str): The name of the dataset.

        Returns:
            tuple: A tuple containing the train and test sets.
        """
        loader_args = {}
        if "data_dir" in kwargs:
            loader_args["data_home"] = kwargs["data_dir"]

        data = getattr(datasets, self.dataset_list[name])(**loader_args)
        target_values = data.target.reshape((-1, 1))
        size = target_values.shape[0]
        self.element_size = data.data.shape[1:]
        self.dataset_size = size

        train_set = tf.data.Dataset.from_tensor_slices(
            (
                data.data[: int(size * self.training_split)],
                target_values[: int(size * self.training_split)],
            )
        )
        test_set = tf.data.Dataset.from_tensor_slices(
            (
                data.data[
                    int(size * self.training_split) : int(
                        size * ((1 + self.training_split) * 0.5)
                    )
                ],
                target_values[
                    int(size * self.training_split) : int(
                        size * ((1 + self.training_split) * 0.5)
                    )
                ],
            )
        )
        eval_set = tf.data.Dataset.from_tensor_slices(
            (
                data.data[int(size * ((1 + self.training_split) * 0.5)) :],
                target_values[int(size * ((1 + self.training_split) * 0.5)) :],
            )
        )
        return (train_set, test_set, eval_set)


def get_datasets(request, pk):
    """
    Retrieves the available datasets from a DatasetLoader object.

    Args:
        request: The HTTP request object.
        pk: The primary key of the DatasetLoader object.

    Returns:
        A JSON response containing the available datasets.

    Raises:
        DatasetLoader.DoesNotExist: If the DatasetLoader object with the given primary key does not exist.
    """
    dataset_loader = DatasetLoader.objects.get(pk=pk)
    available_datasets = dataset_loader.get_datasets()
    return JsonResponse(available_datasets, safe=False)
