import importlib

from loguru import logger


def get_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    class_instance = getattr(module, class_name)
    return class_instance


def get_object(
    module_name: str, class_name: str, additional_arguments, required_arguments=[]
):
    try:
        module = importlib.import_module(module_name)
        class_instance = getattr(module, class_name)
        return class_instance(
            **get_arguments_as_dict(additional_arguments, required_arguments)
        )
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
    except Exception as e:
        logger.critical(
            f"Importing Callback {callback_definition['class_name']} from module {callback_definition['module_name']} failed: {e}"
        )
        return class_instance


def get_arguments_as_dict(additional_arguments, required_arguments):
    arguments = {}
    for argument in additional_arguments:
        if argument["value"] != "undefined":
            # check for dtype:
            arguments[argument["name"]] = argument["value"]
            for required_arg in required_arguments:
                if argument["name"] == required_arg["name"]:
                    if required_arg["dtype"] == "int":
                        try:
                            if is_int(argument["value"]):
                                arguments[argument["name"]] = int(argument["value"])
                            else:
                                arguments[argument["name"]] = float(argument["value"])
                        except Exception as e:
                            logger.error(
                                f"Fehler: Parameter {argument['name']} muss als Zahl gegeben sein: {e}"
                            )
                    elif required_arg["dtype"] == "float":
                        try:
                            arguments[argument["name"]] = float(argument["value"])
                        except Exception as e:
                            logger.error(
                                f"Fehler: Parameter {argument['name']} muss als Kommazahl gegeben sein: {e}"
                            )
                    elif required_arg["dtype"] == "bool":
                        if argument["value"] == "true":
                            argument["value"] = True
                        elif argument["value"] == "false":
                            argument["value"] = False
                        else:
                            try:
                                arguments[argument["name"]] = bool(argument["value"])
                            except Exception as e:
                                logger.error(
                                    f"Fehler: Parameter {argument['name']} muss als Boolean gegeben sein: {e}"
                                )
                    continue

    return arguments


def is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False
