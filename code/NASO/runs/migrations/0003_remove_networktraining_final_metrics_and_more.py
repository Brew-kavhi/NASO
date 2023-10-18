# Generated by Django 4.2.5 on 2023-09-23 09:15

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0002_alter_networktraining_final_metrics"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="networktraining",
            name="final_metrics",
        ),
        migrations.AddField(
            model_name="networktraining",
            name="final_metrics",
            field=models.ForeignKey(
                default=1,
                on_delete=django.db.models.deletion.CASCADE,
                to="runs.trainingmetric",
            ),
            preserve_default=False,
        ),
    ]
