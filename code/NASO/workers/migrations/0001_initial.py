# Generated by Django 4.2.10 on 2024-03-12 23:47

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="CeleryWorker",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("queue_name", models.CharField(max_length=100)),
                ("hostname", models.CharField(max_length=100, unique=True)),
                ("last_ping", models.DateTimeField(auto_now=True)),
                ("devices", models.JSONField(default=dict)),
                ("active", models.BooleanField()),
            ],
        ),
    ]
