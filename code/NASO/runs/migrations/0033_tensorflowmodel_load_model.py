# Generated by Django 4.2.10 on 2024-05-11 11:16

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0032_networktraining_tensorflow_model_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="tensorflowmodel",
            name="load_model",
            field=models.BooleanField(default=False),
        ),
    ]
