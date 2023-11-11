# Generated by Django 4.2.7 on 2023-11-09 15:15

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("neural_architecture", "0033_kerasmodel_alter_autokerasrun_dataset_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="networkconfiguration",
            name="enable_quantization",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="networkconfiguration",
            name="model_file",
            field=models.CharField(default="", max_length=100),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="networkconfiguration",
            name="save_model",
            field=models.BooleanField(default=False),
        ),
        migrations.CreateModel(
            name="PruningScheduleTypes",
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
                ("name", models.CharField(max_length=100)),
                ("required_arguments", models.JSONField(null=True)),
                ("native_pruning_schedule", models.BooleanField(default=False)),
            ],
            options={
                "abstract": False,
                "unique_together": {("module_name", "name")},
            },
        ),
        migrations.CreateModel(
            name="PruningSchedule",
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
                    "instance_type",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="neural_architecture.pruningscheduletypes",
                    ),
                ),
            ],
            options={
                "abstract": False,
            },
        ),
        migrations.CreateModel(
            name="PruningPolicyTypes",
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
                ("name", models.CharField(max_length=100)),
                ("required_arguments", models.JSONField(null=True)),
                ("native_pruning_policy", models.BooleanField(default=False)),
            ],
            options={
                "abstract": False,
                "unique_together": {("module_name", "name")},
            },
        ),
        migrations.CreateModel(
            name="PruningPolicy",
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
                    "instance_type",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="neural_architecture.pruningpolicytypes",
                    ),
                ),
            ],
            options={
                "abstract": False,
            },
        ),
        migrations.CreateModel(
            name="PruningMethodTypes",
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
                ("name", models.CharField(max_length=100)),
                ("required_arguments", models.JSONField(null=True)),
                ("native_pruning_method", models.BooleanField(default=False)),
            ],
            options={
                "abstract": False,
                "unique_together": {("module_name", "name")},
            },
        ),
        migrations.CreateModel(
            name="PruningMethod",
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
                    "instance_type",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="neural_architecture.pruningmethodtypes",
                    ),
                ),
            ],
            options={
                "abstract": False,
            },
        ),
        migrations.AddField(
            model_name="networkconfiguration",
            name="pruning_method",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="neural_architecture.pruningmethod",
            ),
        ),
        migrations.AddField(
            model_name="networkconfiguration",
            name="pruning_policy",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="neural_architecture.pruningpolicy",
            ),
        ),
        migrations.AddField(
            model_name="networkconfiguration",
            name="pruning_schedule",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="neural_architecture.pruningschedule",
            ),
        ),
    ]