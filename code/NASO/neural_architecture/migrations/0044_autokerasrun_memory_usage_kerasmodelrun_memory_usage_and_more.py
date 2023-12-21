# Generated by Django 4.2.8 on 2023-12-20 14:22

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("neural_architecture", "0043_autokerasrun_energy_measurements_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="autokerasrun",
            name="memory_usage",
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="kerasmodelrun",
            name="memory_usage",
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name="autokerasrun",
            name="naso_app_version",
            field=models.CharField(default="0.4.6", max_length=10),
        ),
        migrations.AlterField(
            model_name="kerasmodelrun",
            name="naso_app_version",
            field=models.CharField(default="0.4.6", max_length=10),
        ),
    ]