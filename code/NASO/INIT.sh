# run the database migrations and fill the ffirst registered layers and activoations and stuff loike that  
poetry run python manage.py makemigrations
poetry run python manage.py migrate
poetry run pytohn manage.py collectstatic --noinput
poetry run python manage.py loadneuralutilities