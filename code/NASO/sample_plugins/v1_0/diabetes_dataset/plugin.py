import os
import zipfile
from os import listdir
from os.path import isfile, join

import kaggle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_architecture.models.dataset import DatasetLoader
from plugins.interfaces.commands import InstallerInterface
from plugins.interfaces.dataset import DatasetLoaderInterface


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
    dataset_list = [
        "Diabetes Dataset (3)",
        "Diabetes Dataset (5)",
        "Diabetes Dataset (7)",
    ]
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
        self.dataset_path = os.path.dirname(os.path.abspath(__file__)) + "/data"
        if not os.path.exists(self.dataset_path):
            raise Exception(f"Dataset can not be found on {self.dataset_path}.")

        if name == "Diabetes Dataset (3)":
            self.element_size = 3
            return self.get_diabetes(3)
        elif name == "Diabetes Dataset (5)":
            self.element_size = 5
            return self.get_diabetes(5)
        elif name == "Diabetes Dataset (7)":
            self.element_size = 7
            return self.get_diabetes(7)
        return (None, None, None)

    def get_diabetes(self, sequence_length=3):
        """
        The dataset is struuctured as follows:
        we have one column named _ts that stores timestamps in d-m-Y h:s:i format
        and we have the value for that time that is stored in _value column as baic integer.
        The training data is given in files that contain ws-training
        """
        files = listdir(self.dataset_path)

        training_df_list = [
            pd.read_csv(join(self.dataset_path, file))
            for file in files
            if isfile(join(self.dataset_path, file))
            and "training" in file
            and file.endswith(".csv")
        ]
        training_df = pd.concat(training_df_list)

        testing_df_list = [
            pd.read_csv(join(self.dataset_path, file))
            for file in files
            if isfile(join(self.dataset_path, file))
            and "testing" in file
            and file.endswith(".csv")
        ]
        testing_df = pd.concat(testing_df_list)

        testing_df, eval_df = train_test_split(testing_df, test_size=0.5, shuffle=False)

        train_dataset = self._prepare_dataset(training_df, sequence_length)
        test_dataset = self._prepare_dataset(testing_df, sequence_length)
        eval_dataset = self._prepare_dataset(eval_df, sequence_length)

        return (train_dataset, test_dataset, eval_dataset)

    def _prepare_dataset(self, df, sequence_length):
        df["_ts"] = pd.to_datetime(df["_ts"], format="%d-%m-%Y %H:%M:%S")
        df = df.sort_values(by="_ts")
        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(df["_value"].iloc[i : i + sequence_length].values)
            y.append(df["_value"].iloc[i + sequence_length])
        X, y = np.array(X), np.array(y)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset

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
