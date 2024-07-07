import os
from functools import reduce
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from datasets.helper_scripts.normalizations import compute_mean_and_std, z_normalize_ds
from naso import settings
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

    def get_info(self, name, *args, **kwargs):
        return {"description": "This is a diabetes dataset from university of Ohio."}

    def get_sample_images(self, dataset_name="", num_examples=9):
        filename = f"{dataset_name}.png"
        static_dir = os.path.join(settings.BASE_DIR, "static/datasets")
        os.makedirs(static_dir, exist_ok=True)
        file_path = os.path.join(static_dir, filename)
        if os.path.exists(file_path):
            return f"datasets/{filename}"
        # 1oad the dataset
        train_dataset, _, _ = self.get_data(dataset_name, False, num_examples, True)

        # Prepare the dataset for iteration
        train_dataset = train_dataset.take(num_examples).batch(num_examples)
        # Create an iterator
        iterator = iter(train_dataset)
        images, labels = next(iterator)

        # Plotting the sample images
        plt.figure(figsize=(10, 10))

        for i in range(num_examples):
            plt.subplot(int(num_examples**0.5), int(num_examples**0.5), i + 1)
            x = images[i].numpy()
            y = labels[i].numpy()
            x_length = len(x)
            y_length = len(y)
            plt.plot(x, color="blue")
            plt.plot(range(x_length, x_length + y_length), y, color="red")
            plt.title(str(i))

        plt.savefig(file_path)
        plt.close()
        return f"datasets/{filename}"

    def get_data(
        self,
        name: str,
        as_supervised: bool,
        num_individuals=None,
        normalize=True,
        *args,
        **kwargs,
    ) -> tuple:
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
            raise Exception(f"Dataset can not be found on system {self.dataset_path}.")

        if name == "Diabetes Dataset (3)":
            self.element_size = (3, 1)
            return self.get_diabetes(3, num_individuals, normalize)
        elif name == "Diabetes Dataset (5)":
            self.element_size = (5, 1)
            return self.get_diabetes(5, num_individuals, normalize)
        elif name == "Diabetes Dataset (7)":
            self.element_size = (7, 1)
            return self.get_diabetes(7, num_individuals, normalize)
        return (None, None, None)

    def get_diabetes(self, sequence_length=3, num_individuals=None, normalize=True):
        """
        The dataset is struuctured as follows:
        we have one column named _ts that stores timestamps in d-m-Y h:s:i format
        and we have the value for that time that is stored in _value column as baic integer.
        The training data is given in files that contain ws-training
        """
        files = listdir(self.dataset_path)
        num_examples = num_individuals

        training_files = [
            file
            for file in files
            if isfile(join(self.dataset_path, file))
            and "training" in file
            and file.endswith(".csv")
        ]
        testing_files = [
            file
            for file in files
            if isfile(join(self.dataset_path, file))
            and "testing" in file
            and file.endswith(".csv")
        ]
        if num_individuals:
            num_examples = 1
            training_files = training_files[:num_individuals]
            testing_files = testing_files[:2]
        training_df_list = [
            pd.read_csv(join(self.dataset_path, file)) for file in training_files
        ]
        testing_df_list = [
            pd.read_csv(join(self.dataset_path, file)) for file in testing_files
        ]

        train_dataset = [
            self._prepare_dataset(
                training_df, sequence_length, num_examples=num_examples
            )
            for training_df in training_df_list
        ]
        test_dataset = [
            self._prepare_dataset(
                testing_df, sequence_length, num_examples=num_examples
            )
            for testing_df in testing_df_list[: len(testing_df_list) // 2]
        ]
        eval_dataset = [
            self._prepare_dataset(eval_df, sequence_length, num_examples=num_examples)
            for eval_df in testing_df_list[len(testing_df_list) // 2 :]
        ]

        train_dataset = reduce(lambda ds1, ds2: ds1.concatenate(ds2), train_dataset)
        test_dataset = reduce(lambda ds1, ds2: ds1.concatenate(ds2), test_dataset)
        eval_dataset = reduce(lambda ds1, ds2: ds1.concatenate(ds2), eval_dataset)
        mean, std = compute_mean_and_std(train_dataset)
        if normalize:
            train_dataset = z_normalize_ds(train_dataset, mean, std)
            test_dataset = z_normalize_ds(test_dataset, mean, std)
            eval_dataset = z_normalize_ds(eval_dataset, mean, std)
        return (train_dataset, test_dataset, eval_dataset)

    def _prepare_dataset(
        self,
        df,
        sequence_length,
        prediction_length=9,
        num_examples=None,
        interpolation="spline",
    ):
        """
        params:
         - interpolation: None, 'spline', 'linear', 'polynomial'
        """
        df["_ts"] = pd.to_datetime(df["_ts"], format="%d-%m-%Y %H:%M:%S")
        df = df.sort_values(by="_ts")
        # Set 'timestamp' as the index
        df.set_index("_ts", inplace=True)

        # Resample to 5-minute intervals and interpolate
        if interpolation:
            df = df.resample("5min", origin="start").interpolate(
                method=interpolation, order=3
            )

        def is_consecutive(timestamps, freq="5min"):
            expected = pd.date_range(
                start=timestamps[0], periods=len(timestamps), freq=freq
            )
            return all(timestamps == expected)

        X, y = [], []
        for i in range(len(df) - sequence_length - prediction_length + 1):
            if interpolation or is_consecutive(df.index[i : i + sequence_length]):
                X.append(df["_value"].iloc[i : i + sequence_length].values)
                y.append(
                    df["_value"]
                    .iloc[i + sequence_length : i + sequence_length + prediction_length]
                    .values
                )
                if num_examples and len(X) >= num_examples:
                    break
        X, y = np.array(X), np.array(y)
        X = X.reshape((-1, sequence_length, 1))
        y = y.reshape((-1, prediction_length, 1))
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
