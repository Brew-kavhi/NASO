# Generated by Django 4.2.7 on 2023-11-09 19:32

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            "neural_architecture",
            "0034_networkconfiguration_enable_quantization_and_more",
        ),
    ]

    operations = [
        migrations.AddField(
            model_name="kerasmodel",
            name="enable_quantization",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="kerasmodel",
            name="pruning_method",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="neural_architecture.pruningmethod",
            ),
        ),
        migrations.AddField(
            model_name="kerasmodel",
            name="pruning_policy",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="neural_architecture.pruningpolicy",
            ),
        ),
        migrations.AddField(
            model_name="kerasmodel",
            name="pruning_schedule",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to="neural_architecture.pruningschedule",
            ),
        ),
        migrations.AddField(
            model_name="networkconfiguration",
            name="load_model",
            field=models.BooleanField(default=False),
        ),
    ]
