# Generated by Django 4.2.10 on 2024-02-18 16:06

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0028_rename_energy_measurements_networktraining_power_measurements"),
    ]

    operations = [
        migrations.AlterField(
            model_name="networktraining",
            name="naso_app_version",
            field=models.CharField(default="1.0.0", max_length=10),
        ),
    ]
