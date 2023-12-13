import ast
import importlib

from loguru import logger


def get_class(module_name: str, class_name: str):
    """
    This function returns a class instance build from the module and the class.

    Args:
        module_name (str): The name of the module.
        class_name (str): The name of the class.

    Returns:
        class_instance: The instance of the class.
    """
    module = importlib.import_module(module_name)
    class_instance = getattr(module, class_name)
    return class_instance


def get_object(
    module_name: str, class_name: str, additional_arguments, required_arguments=None
):
    """
    This function returns a class instance build from the module and the class and instantiated with given arguments.

    Args:
        module_name (str): The name of the module.
        class_name (str): The name of the class.
        additional_arguments (list): A list of arguments passed to the constructor.
        required_arguments (list): A list of required arguments.

    Returns:
        class_instance: The instance of the class.
    """
    if required_arguments is None:
        required_arguments = []
    try:
        module = importlib.import_module(module_name)
        class_instance = getattr(module, class_name)
        return class_instance(
            **get_arguments_as_dict(additional_arguments, required_arguments)
        )
    except ImportError as _e:
        logger.critical(
            f"Importing Class {class_name} from module {module_name} failed: {_e}"
        )
        return class_instance()
    except Exception as _e:
        logger.critical(
            f"Immporting Class {class_name} from module {module_name} failed: {_e}"
        )
        return class_instance()


def get_callback(callback_definition, required_arguments=None):
    """
    This function returns a callback instance build from the module and the class and instantiated with given arguments.

    Args:
        callback_definition (dict): A dictionary containing the module_name, class_name and additional_arguments.
        required_arguments (list): A list of required arguments.

    Returns:
        class_instance: The instance of the class.
    """
    if required_arguments is None:
        required_arguments = []
    try:
        module = importlib.import_module(callback_definition["module_name"])
        class_instance = getattr(module, callback_definition["class_name"])
        return class_instance(
            **get_arguments_as_dict(
                callback_definition["additional_arguments"], required_arguments
            )
        )
    except Exception as _e:
        logger.critical(
            f"Importing Callback {callback_definition['class_name']} from "
            + f"module {callback_definition['module_name']} failed: {_e}"
        )
        return class_instance


def get_arguments_as_dict(additional_arguments, required_arguments):
    """
    This function build and retrusn a dictionary of arguments from the given additional arguments.


    Args:
        additional_arguments (list): A list of arguments passed to the constructor.
        required_arguments (list): A list of required arguments.

    Returns:
        dict: A dictionary of arguments.
    """
    arguments = {}
    for argument in additional_arguments:
        if (
            argument["value"]
            and argument["value"] != "undefined"
            and argument["value"] != "None"
            and argument["value"] != "null"
            and argument["value"] != ""
        ):
            # check for dtype:
            arguments[argument["name"]] = argument["value"]
            for required_arg in required_arguments:
                if argument["name"] == required_arg["name"]:
                    build_argument(argument, required_arg, arguments)
                    continue

    return arguments


def build_argument(argument, required_argument, arguments):
    """
    This function builds an argument from the given argument and required argument.

    Args:
        argument (dict): A dictionary containing the name and value of the argument.
        required_argument (dict): A dictionary containing the name and dtype of the argument.
        arguments (dict): A dictionary of arguments.

    Returns:
        dict: A dictionary of arguments.
    """
    try:
        if required_argument["dtype"] == "int":
            if is_int(argument["value"]):
                arguments[argument["name"]] = int(argument["value"])
            else:
                arguments[argument["name"]] = float(argument["value"])

        elif required_argument["dtype"] == "float":
            arguments[argument["name"]] = float(argument["value"])

        elif required_argument["dtype"] == "bool":
            if argument["value"].lower() == "true":
                arguments[argument["name"]] = True
            elif argument["value"].lower() == "false":
                arguments[argument["name"]] = False
            else:
                arguments[argument["name"]] = bool(argument["value"])
        elif required_argument["dtype"].startswith("tuple"):
            # this is a tuple
            arguments[argument["name"]] = ast.literal_eval(argument["value"])
        elif (
            required_argument["dtype"] == "str"
            or required_argument["dtype"] == "NoneType"
        ):
            if argument["value"].lower() == "true":
                arguments[argument["name"]] = True
            elif argument["value"].lower() == "false":
                arguments[argument["name"]] = False
            else:
                arguments[argument["name"]] = argument["value"]
        else:
            arguments[argument["name"]] = ast.literal_eval(argument["value"])
    except ValueError as exc:
        logger.error(
            f"Fehler: Parameter {argument['name']} muss fur {required_argument['dtype']} als Zahl gegeben sein: {exc}"
        )


def is_int(string):
    """
    This function checks if the given string is an integer.

    Args:
        string (str): The string to check.

    Returns:
        bool: True if the string is an integer, False otherwise.
    """
    try:
        int(string)
        return True
    except ValueError:
        return False
