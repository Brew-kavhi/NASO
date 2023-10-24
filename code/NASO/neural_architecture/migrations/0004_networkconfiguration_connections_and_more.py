# Generated by Django 4.2.5 on 2023-10-14 13:49

import django.db.models.deletion
from django.db import migrations, models

import neural_architecture.validators


class Migration(migrations.Migration):
    dependencies = [
        (
            "neural_architecture",
            "0003_rename_keras_natiive_kayeer_networklayertype_keras_native_layer",
        ),
    ]

    operations = [
        migrations.AddField(
            model_name="networkconfiguration",
            name="connections",
            field=models.JSONField(default={}),
        ),
        migrations.AddField(
            model_name="networkconfiguration",
            name="size",
            field=models.IntegerField(default=0),
        ),
        migrations.CreateModel(
            name="NetworkLayer",
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
                ("trainable", models.BooleanField(default=True)),
                ("name", models.CharField(max_length=60)),
                (
                    "dtype",
                    models.CharField(
                        max_length=20,
                        validators=[neural_architecture.validators.validate_dtype],
                    ),
                ),
                ("dynamic", models.BooleanField(default=False)),
                (
                    "layer_type",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.DO_NOTHING,
                        to="neural_architecture.networklayertype",
                    ),
                ),
            ],
            options={
                "abstract": False,
            },
        ),
        migrations.AlterField(
            model_name="networkconfiguration",
            name="layers",
            field=models.ManyToManyField(to="neural_architecture.networklayer"),
        ),
    ]
