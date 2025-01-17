NASO
===
NAS stands for neural architecture search and optimization. It offers an easy way tp configure and conduct neural architecture search and optimzation using a graph basd language to define the search space. This can all be done using a grpahical user interface, naely a webapp coded in django. 

# About
This codebase provides a django webapp to easily configure and run neural networks in tensorflow/keras. You have mutliple options, namely solo runs or autokeras search, which searches for a suited network architecture on its own. Furthermore, the results (the metrics) are displayed in nice graphs to simplify and speed up research processes. Of course, this data can be downloaded and the graphs can be adjusted to your requirements. In addition to that, the system is extendable through plugins. This feature however is not fully implemented yet and needs further adjustments to allow for more integrations.

# Installation
For this project we have several system dependencies, like poetry and GPU toolkit. This chapter is a collection (might be incomplete though as of writing time, 21.12.2023) of system dependencies I know about.
NOte that for python packages we use poetry which allows for easy managemenet of your virtual environment. It collects all the necessary packages in a pyroject.toml file, which is the modern standard for packaging pyhon libraries or applications

## System dependencies
In order to make full use of the gpu you need to install some nvidia-libraries:
```apt install nvidia-cuda-toolkti```
On some systems (student workstations at university) it might already be installed and therefor can be loaded with the command ```module load cuda```
Furthermore, you need poetry installed.
```apt install poetry```
Poetry handles the virtual environnment, and makes it easy to administrate it.

Then clone this project:
```git clone (repo-url)```
````cd code/NASO``
Next you need to install all the dependencies with poetry:
```poetry install```

You may need to install tensorflow for gpu and autokeras separately by executing:
```poetry run pip intall tensorflow[and-cuda] ```
For async task io we use celery. This is automatically installed with poetry, but for the backend and message quieing, we need to install an additional server, namely a rabbitmq server (optional, see below). This handles all our task states and everything. Install the message queue with the following command:
```sudo apt-get install rabbitmq-server```

If its not possible to start or install a rabbitmq server. you can ask me (Marius Goehring) for an alternative. This server does not need to run locally, so you can use any other **hosted rabbitmq instance**. I have one running for this purpose.

Another thing is the measurement of the power consumption. For GPUs this tool uses ```nvidia-smi``` and for CPU we use ```turbostat``` which can be installed by ```apt install linux-cpupower```. These command line tools are invoked in python by spawning a subprocess

## Configuration

Last step is to configure the environment. This roject uses python-decouple to laod environment variables into the django app. This is important to keep sensitive information local and to have all the config in one place. You can find a sample configuration in the file .envrc. just copy it to a file named .env and adjust it to your needs. Here you also need to configure the rabbitmq server instance. This env file also holds the database connection credentials. This is only necessary if you are using something else than sqlite, which is preconfigured in the project settings. So if you are using SQLite, because you might not be able to install a proper SQL-database server, you need to change the database settings in the file naso/settings_local.py. 

For rating the runs there exists an api call. This api call is protected by TokenAuthentication. Therefor you must create an API_TOKEN. This can be done by executing the javascript code from inside the helper_scripts/get_token.html in a console on a browser window of opened NASO. Keep in mind, that therefor you have to create an superuser and set the username and password accordingly in the authentication call. Furthmore, the received token ahs to be adde to the environment variabnles files .env.

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

Furthermore, there need to be two workers running. One is for actually executing the network, the other one is subscribed to a special queue and only handles the loading of models. This needs to be done in a celery task as when laoding the model, the graphics memory is loaded full and because of tensorflows memory leak we have no option of freeing up the memory. The only option is to spawn a subprocess that is killed afterwards. 
You need to start the workers with this command: 
```poetry run celery -A naso worker -l INFO --concurrency 1 -n <name>@%h```
and the following one for the trial loading:
```poetry run celery -A naso worker -Q start_trials -l INFO```
The concurrency parameter ensures that only one task is executed at a time, because otherwise it could lead to weird behaviour.
If you want to daemonize it, so it starts one every reboot, you can find  more information about it in the celery folder of this project. Now if this server is running on a remote machine that is behind another network, like it is the case with the workstations at IPVS, we need to enable port forwarding. this is done with the following command, to be executedon your home desktop: 
```ssh -L <local_port>:<remote-machine>:<django-port> <username>@ipvslogin.informatik.uni-stuttgart.de```
This forwards the local port to the port of the remote machine. Now you also need to make the local system listen to incoming request and forward them to this adress. I therefor use this command:
```socat TCP-LISTEN:<open-local-port>,fork TCP:localhost:<local-port>```
# Documentation
Add this. We use networkX as the tool to create graphs that describe the networks and search space.

# Development
Run 
```poetry run isort .``` for sorting the imports
```poetry run black .``` for formatting and 
```poetry run djlint . --extension=html --reformat```
```poety run flake8 .``` will do a basic code check
For enhanced testing and pep8 style enforcement, you can execute ```poetry run pylint *```. Note that this still, throws a lot of errors and warnings, which will be fixed in future commits.

Django tests are still to be implemented.
