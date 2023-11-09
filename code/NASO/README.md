NASO
===
NAS stands for neural architecture search and optimization. It offers an easy way tp configure and conduct neural architecture search and optimzation using a graph basd language to define the search space. This can all be done using a grpahical user interface, naely a webapp coded in django. 

# About
This codebase provides a django webapp to easily configure and run neural networks in tensorflow/keras. You have mutliple options, namely solo runs or autokeras search, which searches for a suited network architecture on its own. Furthermore, the results (the metrics) are displayed in nice graphs to simplify and speed up research processes. Of course, this data can be downloaded and the graphs can be adjusted to your requirements. In addition to that, the system is extendable through plugins. This feature however is not fully implemented yet and needs further adjustments to allow for more integrations.

# Installation
You need poetry installed.
```apt install poetry```
Poetry handles the virtual environnment, and makes it easy to administrate it.

Then clone this project:
```git clone (repo-url)```
````cd code/NASO``
Next you need to install all the dependencies with poetry:
```poetry install```

You may need to install tensorflow and autokeras separately by executing:
```poetry run pip intall .```
For async task io we use celery. This is automatically installed with poetry, but for the backend and message quieing, we need to install an additional server, namely a rabbitmq server (optional, see below). This handles all our task states and everything. Install the message queue with the following command:
```sudo apt-get install rabbitmq-server```

If its not possible to start or install a rabbitmq server. you can ask me (Marius Goehring) for an alternative. This server does not need to run locally, so you can use any other **hosted rabbitmq instance**. I have one running for this purpose.

## Configuration

Last step is to configure the environment. This roject uses python-decouple to laod environment variables into the django app. This is important to keep sensitive information local and to have all the config in one place. You can find a sample configuration in the file .envrc. just copy it to a file named .env and adjust it to your needs. Here you also need to configure the rabbitmq server instance. This env file also holds the database connection credentials. This is only necessary if you are using something else than sqlite, which is preconfigured in the project settings. So if you are using SQLite, because you might not be able to install a proper SQL-database server, you need to change the database settings in the file naso/settings_local.py. 

# Execution
If you are starting it for the first time, you need to migrate the database:
```poetry run python manage.py migrate```
After that you need to load all the available layers into the database:
```poetry run python manage.py loadneuralutilities```
If you want to have an admin account for more detailed insights, you can create an admin account by exexcuting:
```poetry run python manage.py createsuperuser```
After that the database is setup and you are ready to start the server
Run
```´poetry run python manage.py runserver```
to start the development server. For using the local settings append ```--settings=naso.settings_local```. Depending on your working environment you may need to configure port forwarding so you can acces the frontend. This is necessary if you execute it on a remote machine. If you want to run it on a production server run 
--settings=naso.settings_local--settings=naso.settings_local```poetry run uvicorn naso.asgi:application --host <host> ---port <port>```
(Development server can only handle a few connections at one time, maybe 5-10 max, while production server can handle way more.)

Furthermore, there needs to be a celery worker running. You can starte one with this command: 
```poetry run celery -A naso worker -l INFO --concurrency 1```
The concurrency parameter ensures that only one task is executed at a time, because otherwise it could lead to weird behaviour.
If you want to daemonize it, so it starts one every reboot, you can find  more information about it in the celery folder of this project. Now if this server is running on a remote machine that is behind another network, like it is the case with the workstations at IPVS, we need to enable port forwarding. this is done with the following command, to be executedon your home desktop: 
```ssh -L <local_port>:<remote-machine>:<django-port> <username>@ipvslogin.informatik.uni-stuttgart.de```
# Documentation
Add this. We use networkX as the tool to create graphs that describe the networks and search space.

# Development
Run 
```poetry run isort .``` for sorting the imports
```poetry run black .``` for formatting and 
```poety run flake8 .``` will do a basic code check
For enhanced testing and pep8 style enforcement, you can execute ```poetry run pylint *```. Note that this still, throws a lot of errors and warnings, which will be fixed in future commits.

Django tests are still to be implemented.
