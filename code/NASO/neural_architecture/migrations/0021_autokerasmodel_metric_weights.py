# Generated by Django 4.2.6 on 2023-10-25 13:03

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            "neural_architecture",
            "0020_autokerasmodel_callbacks_autokerasmodel_loss_and_more",
        ),
    ]

    operations = [
        migrations.AddField(
            model_name="autokerasmodel",
            name="metric_weights",
            field=models.JSONField(null=True),
        ),
    ]