# Generated by Django 4.2.6 on 2023-10-18 18:50

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="fitparameters",
            name="verbose",
            field=models.BinaryField(default=1, max_length=2),
        ),
    ]
