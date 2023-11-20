from abc import ABC, abstractmethod


class DatasetLoaderInterface(ABC):
    @abstractmethod
    def get_data(
        self,
        name="",
        is_supervised=True,
        data_dir="",
        shuffle_files=True,
        *args,
        **kwargs
    ):
        pass

    @abstractmethod
    def get_element_size(self, *args, **kwargs):
        """
        retruns the size of an element in this dataset.
        """

    @abstractmethod
    def get_size(self, *args, **kwargs):
        """
        retruns the size of this dataset.
        """

    @abstractmethod
    def get_datasets(self, *args, **kwargs):
        pass
