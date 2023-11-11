import ast
import importlib

from loguru import logger


def get_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    class_instance = getattr(module, class_name)
    return class_instance


def get_object(
    module_name: str, class_name: str, additional_arguments, required_arguments=None
):
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
    arguments = {}
    for argument in additional_arguments:
        if argument["value"] != "undefined" and argument["value"] != "None":
            # check for dtype:
            arguments[argument["name"]] = argument["value"]
            for required_arg in required_arguments:
                if argument["name"] == required_arg["name"]:
                    build_argument(argument, required_arg, arguments)
                    continue

    return arguments


def build_argument(argument, required_argument, arguments):
    if argument["value"] == "" or not argument["value"]:
        return
    if required_argument["dtype"] == "int":
        try:
            if is_int(argument["value"]):
                arguments[argument["name"]] = int(argument["value"])
            else:
                arguments[argument["name"]] = float(argument["value"])
        except ValueError as _e:
            logger.error(
                f"Fehler: Parameter {argument['name']} muss als Zahl gegeben sein: {_e}"
            )
    elif required_argument["dtype"] == "float":
        try:
            arguments[argument["name"]] = float(argument["value"])
        except ValueError as _e:
            logger.error(
                f"Fehler: Parameter {argument['name']} muss als Kommazahl gegeben sein: {_e}"
            )
    elif required_argument["dtype"] == "bool":
        if argument["value"] == "true":
            argument["value"] = True
        elif argument["value"] == "false":
            argument["value"] = False
        else:
            try:
                arguments[argument["name"]] = bool(argument["value"])
            except ValueError as _e:
                logger.error(
                    f"Fehler: Parameter {argument['name']} muss als Boolean gegeben sein: {_e}"
                )
    elif required_argument["dtype"].startswith("tuple"):
        # this is a tuple
        try:
            arguments[argument["name"]] = ast.literal_eval(argument["value"])
        except ValueError as _e:
            logger.error(
                f"Fehler: Parameter {argument['name']} muss als Tuple gegeben sein: {_e}"
            )


def is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False
