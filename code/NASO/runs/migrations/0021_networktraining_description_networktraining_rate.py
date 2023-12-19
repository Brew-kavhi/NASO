# Generated by Django 4.2.7 on 2023-11-30 12:27

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0020_networktraining_gpu"),
    ]

    operations = [
        migrations.AddField(
            model_name="networktraining",
            name="description",
            field=models.TextField(default=""),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="networktraining",
            name="rate",
            field=models.IntegerField(default=0),
        ),
    ]