# Generated by Django 4.2.10 on 2024-05-08 13:39

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("neural_architecture", "0054_tensorflowmodeltype"),
        ("runs", "0030_networktraining_worker"),
    ]

    operations = [
        migrations.CreateModel(
            name="TensorFlowModel",
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
                        on_delete=django.db.models.deletion.DO_NOTHING,
                        to="neural_architecture.tensorflowmodeltype",
                    ),
                ),
            ],
            options={
                "abstract": False,
            },
        ),
    ]