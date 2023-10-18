# Generated by Django 4.2.6 on 2023-10-18 19:21

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0003_remove_fitparameters_verbose"),
    ]

    operations = [
        migrations.AddField(
            model_name="networktraining",
            name="state",
            field=models.CharField(
                choices=[("s", "Gestartet"), ("r", "Lauft"), ("f", "Beendet")],
                default="s",
                max_length=1,
            ),
        ),
    ]
