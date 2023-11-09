from abc import ABC, abstractmethod

from plugins.models.plugins import Plugin


class InstallerInterface(ABC):
    plugin: Plugin
    module_name_prefix = "media.plugins"
    module_name = ""

    def __init__(self, plugin: Plugin):
        self.plugin = plugin
        self.module_name = (
            self.plugin.name
            + "."
            + self.plugin.python_file.name.split("/")[-1].split(".")[0]
        )

    def get_module_name(self):
        return f"{self.module_name_prefix}.{self.module_name}"

    @abstractmethod
    def install(self):
        """
        This method installs the plugin.
        """

    @abstractmethod
    def uninstall(self):
        """
        Uninstalls the plugin and removes all associated files and configurations.
        """
