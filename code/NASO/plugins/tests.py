import json
from io import BytesIO
from unittest.mock import MagicMock, patch

from django.contrib import messages
from django.test import TestCase
from django.urls import reverse

from plugins.forms.plugin_form import PluginForm
from plugins.models.plugins import Plugin


class NewPluginTestCase(TestCase):
    def setUp(self):
        self.url = reverse("plugins:new")
        self.plugin_data = {
            "name": "test_plugin",
            "author": "test_author",
            "description": "test_description",
            "version": "1.0.0",
        }
        self.plugin_files = {
            "python_file": BytesIO(b"print('Hello, world!')"),
            "config_file": BytesIO(json.dumps(self.plugin_data).encode("utf-8")),
        }
        self.request = MagicMock()
        self.request.POST = self.plugin_data
        self.request.FILES = self.plugin_files

    def test_get(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "plugins/upload_plugin.html")
        self.assertIsInstance(response.context["form"], PluginForm)

    @patch("plugins.views.new_plugin.install_plugin")
    @patch("plugins.views.new_plugin.uninstall_plugin")
    def test_post(self, mock_uninstall_plugin, mock_install_plugin):
        response = self.client.post(self.url, self.plugin_data, self.plugin_files)
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("plugins:list"))
        plugin = Plugin.objects.get(name=self.plugin_data["name"])
        self.assertEqual(plugin.author, self.plugin_data["author"])
        self.assertEqual(plugin.description, self.plugin_data["description"])
        self.assertEqual(plugin.version, self.plugin_data["version"])
        self.assertEqual(
            plugin.python_file.read(), self.plugin_files["python_file"].getvalue()
        )
        self.assertEqual(
            plugin.config_file.read(), self.plugin_files["config_file"].getvalue()
        )
        mock_install_plugin.assert_called_once_with(plugin)
        self.assertEqual(messages.SUCCESS, self.request._messages.messages[0].level)
        self.assertEqual(
            f"Plugin {plugin} wurde erfolgreich installiert.",
            self.request._messages.messages[0].message,
        )
        mock_uninstall_plugin.assert_called_once_with(plugin)

    def test_post_invalid_form(self):
        self.plugin_data["version"] = "invalid_version"
        response = self.client.post(self.url, self.plugin_data, self.plugin_files)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "plugins/upload_plugin.html")
        self.assertIsInstance(response.context["form"], PluginForm)
        self.assertIn("version", response.context["form"].errors)
        self.assertFalse(Plugin.objects.filter(name=self.plugin_data["name"]).exists())
