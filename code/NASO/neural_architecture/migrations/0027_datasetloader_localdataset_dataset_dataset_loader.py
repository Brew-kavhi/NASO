# Generated by Django 4.2.6 on 2023-10-27 19:30

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            "neural_architecture",
            "0026_alter_dataset_name_alter_dataset_unique_together",
        ),
    ]

    operations = [
        migrations.CreateModel(
            name="DatasetLoader",
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
                ("module_name", models.CharField(max_length=255)),
                ("class_name", models.CharField(max_length=64)),
                ("name", models.CharField(max_length=64)),
                ("description", models.TextField()),
            ],
            options={
                "unique_together": {("class_name", "module_name")},
            },
        ),
        migrations.CreateModel(
            name="LocalDataset",
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
                ("file", models.FileField(upload_to="datasets/")),
                ("name", models.CharField(max_length=64)),
                ("remote_source", models.CharField(max_length=255, null=True)),
                (
                    "dataloader",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to="neural_architecture.datasetloader",
                    ),
                ),
            ],
        ),
        migrations.AddField(
            model_name="dataset",
            name="dataset_loader",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to="neural_architecture.datasetloader",
            ),
        ),
    ]