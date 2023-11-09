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
    def get_datasets(self, *args, **kwargs):
        pass
