# Generated by Django 4.2.7 on 2023-11-08 15:25

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0014_alter_networktraining_naso_app_version"),
        ("neural_architecture", "0032_alter_autokerasmodel_directory_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="KerasModel",
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
                ("model_file", models.CharField(max_length=100)),
                ("name", models.CharField(max_length=200)),
                ("description", models.CharField(max_length=200)),
                ("size", models.IntegerField(default=-1)),
                (
                    "evaluation_parameters",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="runs.evaluationparameters",
                    ),
                ),
                (
                    "fit_parameters",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="runs.fitparameters",
                    ),
                ),
                ("metrics", models.ManyToManyField(to="runs.metric")),
            ],
        ),
        migrations.AlterField(
            model_name="autokerasrun",
            name="dataset",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="neural_architecture.dataset",
            ),
        ),
        migrations.AlterField(
            model_name="autokerasrun",
            name="naso_app_version",
            field=models.CharField(default="0.1.2", max_length=10),
        ),
        migrations.CreateModel(
            name="KerasModelRun",
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
                ("naso_app_version", models.CharField(default="0.1.2", max_length=10)),
                ("git_hash", models.CharField(blank=True, max_length=40)),
                (
                    "dataset",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to="neural_architecture.dataset",
                    ),
                ),
                (
                    "metrics",
                    models.ManyToManyField(
                        related_name="kerasmodel_metrics", to="runs.trainingmetric"
                    ),
                ),
                (
                    "model",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="neural_architecture.kerasmodel",
                    ),
                ),
            ],
            options={
                "abstract": False,
            },
        ),
    ]