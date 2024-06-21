# from django.db imort transaction
import os
import zipfile

import kaggle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from naso import settings
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
                class_name="CaliforniaHousingDataset",
                name="Kaggle Housing Datasets",
                description="These are all the datasets that are available in sklearn.",
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


class CaliforniaHousingDataset(DatasetLoaderInterface):
    """
    A class for loading datasets from kaggle by url

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

    module_name = "kaggle_datasets"
    dataset_path = "datasets/kaggle"
    dataset_list = ["California Housing", "DeepSat (SAT-6)"]
    size = 0
    element_size = 0

    def get_info(self, name, *args, **kwargs):
        datasets_found = kaggle.api.dataset_list(search=name)
        if len(datasets_found) == 0:
            raise Exception("Dataset can not be found on Kaggle")

        dataset = datasets_found[0]
        return vars(dataset)

    def get_data(self, name: str, as_supervised: bool, *args, **kwargs) -> tuple:
        """
        Loads and preprocesses the specified dataset.

        Args:
            name (str): The name of the dataset to load.
            as_supervised (bool): Whether to load the dataset in supervised mode.

        Returns:
            tuple: A tuple containing the train and test datasets.
        """
        datasets_found = kaggle.api.dataset_list(search=name)
        if len(datasets_found) == 0:
            raise Exception("Dataset can not be found on Kaggle")

        dataset = datasets_found[0]
        self.dataset_path += str(dataset.id)
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
            print("Downloading dataset files")
            kaggle.api.dataset_download_files(
                dataset.ref, path=self.dataset_path, unzip=True
            )

        if name == "California Housing":
            self.element_size = 9
            return self.get_california_housing()
        elif name == "DeepSat (SAT-6)":
            self.element_size = (28, 28, 4)
            return self.get_deepsat()
        return (None, None, None)

    def get_california_housing(self):
        df = pd.read_csv(self.dataset_path + "/housing.csv")

        (training_set, test_set, eval_set) = self._prep_dataframe(df)

        train_dataset = tf.data.Dataset.from_tensor_slices(training_set)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_set)
        eval_dataset = tf.data.Dataset.from_tensor_slices(eval_set)

        return (train_dataset, test_dataset, eval_dataset)

    def get_sample_images(self, dataset_name="", num_examples=9):
        if dataset_name == "DeepSat (SAT-6)":
            filename = "deepsat.png"
            static_dir = os.path.join(settings.BASE_DIR, "static/datasets")
            os.makedirs(static_dir, exist_ok=True)
            file_path = os.path.join(static_dir, filename)
            if os.path.exists(file_path):
                return f"datasets/{filename}"

            self.dataset_path = "datasets/kaggle9971"
            train_dataset, _, _ = self.get_deepsat()
            # Prepare the dataset for iteration
            train_dataset = train_dataset.take(num_examples).batch(num_examples)

            # Create an iterator
            info = pd.read_csv(self.dataset_path + "/sat6annotations.csv", header=None)
            annotation = []
            for i, row in info.iterrows():
                annotation.append(row.iloc[0])
            iterator = iter(train_dataset)
            images, labels = next(iterator)
            plt.figure(figsize=(10, 10))

            for i in range(num_examples):
                plt.subplot(int(num_examples**0.5), int(num_examples**0.5), i + 1)
                rgb_image = images[i][:, :, :3].numpy().astype("uint8")
                plt.imshow(rgb_image)
                plt.title(str(annotation[np.argmax(labels[i].numpy())]))
                plt.axis("off")

            plt.savefig(file_path)
            plt.close()
            return f"datasets/{filename}"
        return ""

    def get_deepsat(self):
        n = 20000
        x_train_df = pd.read_csv(
            self.dataset_path + "/X_train_sat6.csv", nrows=n, header=None
        )
        x_train = (
            x_train_df.values.reshape((-1, 28, 28, 4)).clip(0, 2555).astype("float32")
        )
        y_train_df = pd.read_csv(
            self.dataset_path + "/y_train_sat6.csv", nrows=n, header=None
        )
        y_train = y_train_df.values.astype("float32")
        x_test_df = pd.read_csv(
            self.dataset_path + "/X_test_sat6.csv", nrows=n * 0.5, header=None
        )
        x_test = (
            x_test_df.values.reshape((-1, 28, 28, 4)).clip(0, 255).astype("float32")
        )
        y_test_df = pd.read_csv(
            self.dataset_path + "/y_test_sat6.csv", nrows=n * 0.5, header=None
        )
        y_test = y_test_df.values.astype("float32")

        x_eval_df = pd.read_csv(
            self.dataset_path + "/X_test_sat6.csv",
            skiprows=int(n * 0.5),
            nrows=n * 0.5,
            header=None,
        )
        x_eval = (
            x_eval_df.values.reshape((-1, 28, 28, 4)).clip(0, 255).astype("float32")
        )
        y_eval_df = pd.read_csv(
            self.dataset_path + "/y_test_sat6.csv",
            skiprows=int(n * 0.5),
            nrows=n * 0.5,
            header=None,
        )
        y_eval = y_eval_df.values.astype("float32")

        self.size = int(x_train.shape[0])

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        eval_dataset = tf.data.Dataset.from_tensor_slices((x_eval, y_eval))

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
