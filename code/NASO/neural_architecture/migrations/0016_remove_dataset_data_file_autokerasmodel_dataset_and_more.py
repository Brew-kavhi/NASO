# Generated by Django 4.2.6 on 2023-10-23 14:23

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("neural_architecture", "0015_alter_autokerasmodel_project_name"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="dataset",
            name="data_file",
        ),
        migrations.AddField(
            model_name="autokerasmodel",
            name="dataset",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="neural_architecture.dataset",
            ),
        ),
        migrations.AddField(
            model_name="dataset",
            name="as_supervised",
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name="dataset",
            name="data_dir",
            field=models.CharField(
                default="/home/studium/Masterarbeit/code/NASO/neural_architecture/templates",
                max_length=100,
                null=True,
            ),
        ),
    ]
