import importlib.util
import json
import os
import shutil
import zipfile

from django.conf import settings
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
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
        self.context["form"] = form
        if form.is_valid():
            f = form.cleaned_data["file"]
            upload_dir = os.path.join(settings.MEDIA_ROOT, "plugins/.cache")
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            fs = FileSystemStorage(location=upload_dir)
            filename = fs.save(f.name, f)
            uploaded_file_path = fs.path(filename)

            # Extract config.json to get plugin name
            config_data = []
            with zipfile.ZipFile(uploaded_file_path, "r") as zip_ref:
                if "config.json" not in zip_ref.namelist():
                    self.context["form"].add_error(
                        "file", "No config.json found in the plugin."
                    )
                    return self.render_to_response(self.context)
                config_content = zip_ref.read("config.json")
                config_data = json.loads(config_content.decode("utf-8"))
                plugin_name = config_data.get("name")

                if not plugin_name:
                    self.context["form"].add_error(
                        "file", "Plugin name not found in config.json."
                    )
                    return self.render_to_response(self.context)

                # Define the directory where the file will be unzipped
                unzip_dir = os.path.join(settings.MEDIA_ROOT, "plugins/" + plugin_name)
                if not os.path.exists(unzip_dir):
                    os.makedirs(unzip_dir)

                # Unzip the file
                zip_ref.extractall(unzip_dir)
                # remove the zip file from cache
                os.remove(uploaded_file_path)

            # Look for main.py and config.json
            setup_file = os.path.join(unzip_dir, "setup.py")
            config_file = os.path.join(unzip_dir, "config.json")

            if not os.path.isfile(setup_file):
                self.context["form"].add_error(
                    "file", "setup.py not found in the zip file."
                )
                shutil.rmtree(unzip_dir)
                return self.render_to_response(self.context)
            if not os.path.isfile(config_file):
                self.context["form"].add_error(
                    "file", "config.json not found in the zip file."
                )
                shutil.rmtree(unzip_dir)
                return self.render_to_response(self.context)

            if Plugin.objects.filter(
                name=plugin_name, version=config_data.get("version")
            ).exists():
                self.context["form"].add_error(
                    "file", "plugin seems to be installed already."
                )
                shutil.rmtree(unzip_dir)
                return self.render_to_response(self.context)

            plugin = Plugin.objects.create(
                name=config_data.get("name"),
                author=config_data.get("author"),
                description=config_data.get("description"),
                version=config_data.get("version"),
                folder_name=unzip_dir,
            )

            # Call the install function from the Python file
            try:
                install_plugin(plugin)
            except Exception as e:
                self.context["form"].add_error("file", str(e))
                remove_plugin(plugin)
                return self.render_to_response(self.context)

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
        "plugin_module", os.path.join(plugin.folder_name, "setup.py")
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


def remove_plugin(plugin):
    shutil.rmtree(plugin.folder_name)
    plugin.delete()


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
        "plugin_module", os.path.join(plugin.folder_name, "setup.py")
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

    # Use shutil.rmtree() to delete the folder and its contents
    remove_plugin(plugin)
