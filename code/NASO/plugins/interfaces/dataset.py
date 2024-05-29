from abc import ABC, abstractmethod


class DatasetLoaderInterface(ABC):
    @abstractmethod
    def get_data(self, name="", *args, **kwargs):
        """
        Retrieves the data from the dataset.

        Args:
            name (str): The name of the dataset.
            is_supervised (bool): Whether the dataset is supervised or not.
            data_dir (str): The directory where the dataset is stored.
            shuffle_files (bool): Whether to shuffle the files in the dataset.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The retrieved data from the dataset.
        """

    @abstractmethod
    def get_element_size(self, *args, **kwargs):
        """
        Returns the size of an element in this dataset.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The size of an element in this dataset.
        """

    @abstractmethod
    def get_size(self, *args, **kwargs):
        """
        Returns the size of this dataset.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The size of this dataset.
        """

    @abstractmethod
    def get_datasets(self, *args, **kwargs):
        """
        Retrieves the datasets.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The retrieved datasets.
        """
