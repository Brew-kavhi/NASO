# Generated by Django 4.2.10 on 2024-02-12 22:58

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("neural_architecture", "0047_autokerasrun_compute_device_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="autokerasmodel",
            name="trial_folder",
            field=models.CharField(default="", max_length=256),
        ),
    ]
