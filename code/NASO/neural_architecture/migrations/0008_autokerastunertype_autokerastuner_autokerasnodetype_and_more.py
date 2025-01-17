# Generated by Django 4.2.6 on 2023-10-19 08:38

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("neural_architecture", "0007_remove_autokerasnode_node_type_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="AutoKerasTunerType",
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
                ("name", models.CharField(max_length=100, unique=True)),
                ("required_arguments", models.JSONField()),
                (
                    "module_name",
                    models.CharField(default=None, max_length=200, null=True),
                ),
                ("native_tuner", models.BooleanField(default=True)),
            ],
            options={
                "abstract": False,
                "unique_together": {("module_name", "name")},
            },
        ),
        migrations.CreateModel(
            name="AutoKerasTuner",
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
                ("additional_arguments", models.JSONField()),
                (
                    "tuner_type",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="neural_architecture.autokerastunertype",
                    ),
                ),
            ],
            options={
                "abstract": False,
            },
        ),
        migrations.CreateModel(
            name="AutoKerasNodeType",
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
                ("module_name", models.CharField(max_length=150)),
                ("name", models.CharField(max_length=100, unique=True)),
                ("required_arguments", models.JSONField()),
                ("keras_type", models.CharField(max_length=100)),
            ],
            options={
                "abstract": False,
                "unique_together": {("module_name", "name")},
            },
        ),
        migrations.CreateModel(
            name="AutoKerasNode",
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
                ("additional_arguments", models.JSONField()),
                (
                    "node_type",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="neural_architecture.autokerasnodetype",
                    ),
                ),
            ],
            options={
                "abstract": False,
            },
        ),
        migrations.CreateModel(
            name="AutoKerasModel",
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
                (
                    "project_name",
                    models.CharField(default="auto_model", max_length=100),
                ),
                ("max_trials", models.IntegerField(default=100)),
                (
                    "directory",
                    models.CharField(default=None, max_length=100, null=True),
                ),
                ("objective", models.CharField(default="val_loss", max_length=100)),
                ("max_model_size", models.IntegerField(null=True)),
                (
                    "inputs",
                    models.ManyToManyField(
                        related_name="Inputs", to="neural_architecture.autokerasnode"
                    ),
                ),
                (
                    "outputs",
                    models.ManyToManyField(
                        related_name="Outputs", to="neural_architecture.autokerasnode"
                    ),
                ),
                (
                    "tuner",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to="neural_architecture.autokerastuner",
                    ),
                ),
            ],
        ),
    ]
