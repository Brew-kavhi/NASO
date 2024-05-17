from abc import ABC, abstractmethod

from plugins.models.plugins import Plugin


class InstallerInterface(ABC):
    """
    The base abstract class for plugin installers.
    """

    plugin: Plugin
    module_name_prefix = "media.plugins"
    module_name = ""

    def __init__(self, plugin: Plugin):
        """
        Initializes the InstallerInterface with the specified plugin.

        Args:
            plugin (Plugin): The plugin to be installed.
        """
        self.plugin = plugin
        self.module_name = self.plugin.name

    def get_module_name(self) -> str:
        """
        Returns the fully qualified module name for the plugin.

        Returns:
            str: The fully qualified module name.
        """
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
