# Generated by Django 4.2.7 on 2023-11-12 12:57

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0016_networktraining_prediction_metrics"),
        ("neural_architecture", "0035_kerasmodel_enable_quantization_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="autokerasrun",
            name="prediction_metrics",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="autokeras_prediction_metrics",
                to="runs.trainingmetric",
            ),
        ),
        migrations.AddField(
            model_name="kerasmodelrun",
            name="prediction_metrics",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="keras_model_prediction_metrics",
                to="runs.trainingmetric",
            ),
        ),
    ]
