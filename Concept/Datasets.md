Datasets
===
# Loader
This class is responsible for loading the dataset, Therefor it needs to know which dataset it should load and a few additional arguments. And it needs to provide methods for detching available datasets so the users know which datasets are available, and it needs a method for actually fetching the data. This is wehre the additional_arguments are requred.
## Fetch datasets
This is the place to laod all the possible datasets. If i ythi
## load a dataset
This is the function where the datest is finally loaded according to the class
So best again write an interface that the DatasetLoaders need to implement w3ith issubclass(loader, LoaderInteface) in the save method of the loadermodel


# Models
## LocalDataset
I need a model to store local datasets with folder and name amd a dataloadr instance and an optional remote_source.

## Loader
this is basically a type of the dataloader. so i need module name and class name and function names for the given loader.
Then it also needs required_loading_arguments for the arguments that needs to be passed to load_dataset function
Furthermore it needs a get_view function to get a form that enables users to create datasets of this type.

## Dataset 
This class holds the additional_arguments as well as the loader class itself. 

# Summary
This way we can add as many dataloaders as we weant by By registering a dataloader. This dataloader is able to fetch all datasets, like tf.list_builders and local_datasets by fethcing entries form loadDatset model. Furthermore the data can be loaded by using the classes logic, for csv loader or whatever is needed