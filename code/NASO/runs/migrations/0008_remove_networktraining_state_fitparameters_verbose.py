# Generated by Django 4.2.6 on 2023-10-18 19:58

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0007_merge_20231018_1956"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="networktraining",
            name="state",
        ),
    ]
