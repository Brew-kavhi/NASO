# Generated by Django 4.2.9 on 2024-01-08 16:02

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0026_networktraining_size_on_disk"),
    ]

    operations = [
        migrations.AddField(
            model_name="networktraining",
            name="compute_device",
            field=models.CharField(default="", max_length=20),
        ),
    ]