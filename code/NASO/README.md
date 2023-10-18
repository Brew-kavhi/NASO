NASO
===
NAS stands for neural architecture search and optimization. It offers an easy way tp configure and conduct neural architecture search and optimzation using a graph basd language to define the search space. This can all be done using a grpahical user interface, naely a webapp coded in django. 

# Installation
You need poetry installed.
```apt install poetry```
Poetry handles the virtual environnment, and makes it easy administrate it.

Then clone this project:
```git clone (repo-url)```
````cd NASO``
Next you need to install all the dependencies with poetry:
```poetry install```
Unfortunately, it is not possible to install tensorflow with poetry. So you need to manually install it:
```poetry run pip install tensorflow```
This installs tensorflow in poetrys virtual environment.

Last but not least, we need autokeras:
```poetry run pip install autokeras```


For async task io we use celery. This is automatically installed with poetry, but for the backend and message quieing, we need to install an additional server, namely a rabbitmq server. This handles all our task states and everything. Install the message queue with the following command:
```sudo apt-get install rabbitmq-server```
Last step is to configure the environment. This roject uses python-decouple to laod environment variables into the django app. This is important to keep sensitive information local and to have all the config in one place. You can find a sample configuration in the file .envrc. ust copy it to a file named .env and adjust it to yiur needs.

# Execution
Run
```´poetry run python manage.py runserver```
to start the development server.
Furthermore, there needs to be a celery worker running. You can starte one with this command: 
```poetry run celery -A naso worker -l INFO```
If you want to daemonize it, so it starts one every reboot, you can find  more information about it in the celery folder of this project.

# Documentation
Add this. We use networkX as the tool to create graphs that describe the networks and search space.