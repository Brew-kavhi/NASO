import importlib

from loguru import logger


def get_object(module_name: str, class_name: str, additional_arguments):
    try:
        module = importlib.import_module(module_name)
        class_instance = getattr(module, class_name)
        return class_instance(**get_arguments_as_dict(additional_arguments))
    except Exception as e:
        logger.critical(
            f"Immporting Class {class_name} from module {module_name} failed: {e}"
        )
        return class_instance()


def get_callback(callback_definition):
    try:
        module = importlib.import_module(callback_definition["module_name"])
        class_instance = getattr(module, callback_definition["class_name"])
        return class_instance(
            **get_arguments_as_dict(callback_definition["additional_arguments"])
        )
    except:
        logger.critical(
            f"Immporting Callback {callback_definition['class_name']} from module {callback_definition['module_name']} failed."
        )
        return class_instance


def get_arguments_as_dict(additional_arguments):
    arguments = {}
    for argument in additional_arguments:
        if not argument['value'] == 'undefined':
            arguments[argument["name"]] = argument["value"]

    return arguments
