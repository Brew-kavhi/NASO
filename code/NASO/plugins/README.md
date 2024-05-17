Plugins
===
A plugin comes in the form of a zip fiole. The zip file neesd to contain two main files:
- setup.py
- config.json

# setup.py
This file holds the Installer command. Nothing more is required. Pay attention to the installer registering the correct class in the database. The classname must be given with respect to the plugin folder.

# Config.json
This is just a configuration file for the plugin. This holds sopme basic information bout the plugin like authors and descuiprtion and other stuff.
