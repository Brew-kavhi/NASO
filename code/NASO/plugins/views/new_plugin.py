import importlib.util
import json
import os
import shutil

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect
from django.views.generic.base import TemplateView
from loguru import logger

from naso.models.page import PageSetup
from plugins.forms.plugin_form import PluginForm
from plugins.models.plugins import Plugin


class NewPlugin(TemplateView):
    """
    A view for uploading a new plugin.

    Attributes:
    - template_name (str): The name of the HTML template to use for rendering the view.
    - page (PageSetup): An instance of the PageSetup class for setting up the page context.
    - context (dict): A dictionary containing the context variables to be passed to the template.

    Methods:
    - get(self, request, *args, **kwargs): Renders the view for a GET request.
    - post(self, request, *args, **kwargs): Handles form submission for a POST request.
    """

    template_name = "plugins/upload_plugin.html"

    page = PageSetup(title="Plugins", description="Neu")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        """
        Renders the view for a GET request.

        Returns:
        - A response object containing the rendered template.
        """
        form = PluginForm()
        self.context["form"] = form
        return self.render_to_response(self.context)

    def post(self, request, *args, **kwargs):
        """
        Handles form submission for a POST request.

        Returns:
        - A response object containing the rendered template, with form errors if the form is invalid.
        - A redirect response to the plugins:list URL if the form is valid.
        """
        form = PluginForm(request.POST, request.FILES)
        if form.is_valid():
            plugin = form.save(commit=False)
            # Create a subdirectory with the plugin's name
            config_file_content = (
                form.cleaned_data["config_file"].read().decode("utf-8")
            )
            config_data = json.loads(config_file_content)
            # Extract name, author, and version from the JSON
            plugin.name = config_data.get("name")
            plugin.author = config_data.get("author")
            plugin.description = config_data.get("description")
            plugin.version = config_data.get("version")
            # Save the plugin files in the subdirectory
            plugin.python_file.name = os.path.join(plugin.name, plugin.python_file.name)
            plugin.config_file.name = os.path.join(plugin.name, plugin.config_file.name)

            plugin.save()
            # Call the install function from the Python file
            install_plugin(plugin)
            messages.add_message(
                request,
                messages.SUCCESS,
                f"Plugin {plugin} wurde erfolgreich installiert.",
            )
            logger.success(f"Plugin {plugin} wurde erfolgreich installiert.")

            # remove old versions of this plugin
            old_plugins = Plugin.objects.filter(
                name=plugin.name, version__lt=plugin.version
            )
            for old_plugin in old_plugins:
                uninstall_plugin(old_plugin)
                logger.success(f"Plugin {old_plugin} wurde entfernt.")

            return redirect("plugins:list")
        self.context["form"] = form
        return self.render_to_response(self.context)


class UninstallPlugin(TemplateView):
    """
    View class for uninstalling a plugin.

    This class handles the GET request for uninstalling a plugin. It retrieves the plugin
    object based on the provided primary key (pk), calls the uninstall_plugin function to
    uninstall the plugin, and then redirects the user to the plugin list page.

    Attributes:
        template_name (str): The name of the template to be rendered.

    Methods:
        get(request, pk, *args, **kwargs): Handles the GET request for uninstalling a plugin.
    """

    template_name = "uninstall_plugin.html"

    def get(self, request, pk, *args, **kwargs):
        plugin = get_object_or_404(Plugin, pk=pk)
        # Call the uninstall function from the Python file
        uninstall_plugin(plugin)

        messages.add_message(
            request,
            messages.SUCCESS,
            f"Plugin {plugin} wurde erfolgreich entfernt.",
        )
        logger.success(f"Plugin {plugin} wurde erfolgreich entfernt.")
        return redirect("plugins:list")


def install_plugin(plugin):
    """
    Install a plugin by dynamically loading and executing its installer.
    Therefor the python file must contain a Installer Class with an install method.

    Args:
        plugin: An instance of the plugin to be installed.

    Returns:
        None
    """
    spec = importlib.util.spec_from_file_location(
        "plugin_module", plugin.python_file.path
    )
    plugin_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin_module)
    if hasattr(plugin_module, "Installer"):
        # Create an instance of the InstallerClass
        installer = plugin_module.Installer(plugin)

        # Call the install method on the installer
        installer.install()
    else:
        print("InstallerClass not found in the loaded module.")


def uninstall_plugin(plugin):
    """
    Uninstalls a plugin by executing the uninstallation process.
    which must be defined in the plugin's Python files Installer class.

    Args:
        plugin: The plugin object to be uninstalled.

    Returns:
        None
    """
    spec = importlib.util.spec_from_file_location(
        "plugin_module", plugin.python_file.path
    )
    plugin_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin_module)
    # Check if the 'uninstall' method exists in the module
    if hasattr(plugin_module, "Installer"):
        # Create an instance of the InstallerClass
        installer = plugin_module.Installer(plugin)

        # Call the install method on the installer
        installer.uninstall()
    else:
        logger.info("Installer not found in the loaded module.")
    # Delete the plugin files
    folder_path = "media/plugins/" + plugin.name + "/"

    # Use shutil.rmtree() to delete the folder and its contents
    shutil.rmtree(folder_path)
    plugin.delete()
