# from django.db imort transaction
import os
import zipfile

import kaggle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from neural_architecture.models.dataset import DatasetLoader
from plugins.interfaces.commands import InstallerInterface
from plugins.interfaces.dataset import DatasetLoaderInterface

# @transaction.atomic


class Installer(InstallerInterface):
    def install(self):
        # Implement the installation logic here
        try:
            # Create instances in the database
            _, _ = DatasetLoader.objects.get_or_create(
                module_name=self.get_module_name(),
                class_name="OhioDiabetesDatasets",
                name="Ohio Diabetes Datasets",
                description="This is the dataset loader for the diabetes dataset provided by ohio university",
            )
            # You can add more database operations or any other setup logic here

            print("Plugin installed successfully.")
        except Exception as e:
            # Handle any exceptions that may occur during installation
            print(f"Plugin installation failed: {str(e)}")
        print("Installation completed")

    def uninstall(self):
        # Implement the uninstallation logic here
        DatasetLoader.objects.filter(
            module_name=self.get_module_name(),
        ).delete()
        print("Uninstallation completed")


class OhioDiabetesDatasets(DatasetLoaderInterface):
    """
    A class for loading the diabetes dataset

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

    module_name = "diabetes_dataset"
    dataset_path = "data"
    dataset_list = ["Diabetes Dataset"]
    size = 0
    element_size = 0
    info: dict = {}

    def get_data(self, name: str, as_supervised: bool, *args, **kwargs) -> tuple:
        """
        Loads and preprocesses the specified dataset.

        Args:
            name (str): The name of the dataset to load.
            as_supervised (bool): Whether to load the dataset in supervised mode.

        Returns:
            tuple: A tuple containing the train and test datasets.
        """
        self.dataset_path += str(dataset.id)
        if not os.path.exists(self.dataset_path):
            raise Exception("Dataset can not be found on system.")

        if name == "Diabetes Dataset":
            self.element_size = 9
            return self.get_diabetes_dataset()
        return (None, None, None)

    def get_diabetes(self):
        df = pd.read_csv(self.dataset_path + "/bloodsugar.csv")

        (training_set, test_set, eval_set) = self._prep_dataframe(df)

        train_dataset = tf.data.Dataset.from_tensor_slices(training_set)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_set)
        eval_dataset = tf.data.Dataset.from_tensor_slices(eval_set)

        return (train_dataset, test_dataset, eval_dataset)

    def _prep_dataframe(self, dataframe: pd.DataFrame):
        dataframe.dropna(inplace=True)
        columns = [0, 1, 2, 3, 4, 5, 6, 7, 9]
        validation_split = 0.8

        # prepare dataframe
        data = dataframe.iloc[:, columns]
        target = dataframe.iloc[:, 8]
        data.iloc[:, 8], mapping = pd.factorize(data.iloc[:, 8])

        # conversion to numpy
        n_data = np.asarray(data.to_numpy()).astype("float32")
        n_target = np.asarray(target.to_numpy()).astype("float32")

        scaler = StandardScaler()
        n_data = scaler.fit_transform(n_data)
        n_target = n_target / np.max(n_target)

        # training and validation split
        self.size = int(n_data.shape[0] * validation_split)
        self.test_size = int(n_data.shape[0] * (validation_split + 0.1))
        training_data = n_data[: self.size]
        test_data = n_data[self.size : self.test_size]
        eval_data = n_data[self.test_size :]

        training_labels = n_target[: self.size]
        test_labels = n_target[self.size : self.test_size]
        eval_labels = n_target[self.test_size :]

        return (
            (training_data, training_labels),
            (test_data, test_labels),
            (eval_data, eval_labels),
        )

    def get_size(self, *args, **kwargs):
        """
        Returns the size information of the dataset.

        Returns:
            tuple: The shape of the dataset splits.
        """
        return self.size

    def get_element_size(self, *args, **kwargs):
        """
        Returns the size information of each element in the dataset.

        Returns:
            tuple: The shape of the dataset elements.

        Raises:
            Exception: If the dataset does not support size information.
        """
        return self.element_size

    def get_datasets(self, *args, **kwargs):
        """
        Returns the list of available datasets.

        Returns:
            list: A list of available datasets in TensorFlow Datasets.
        """
        return self.dataset_list
