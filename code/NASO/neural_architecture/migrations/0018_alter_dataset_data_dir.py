# Generated by Django 4.2.6 on 2023-10-23 19:22

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("neural_architecture", "0017_remove_autokerasmodel_dataset_autokerasrun"),
    ]

    operations = [
        migrations.AlterField(
            model_name="dataset",
            name="data_dir",
            field=models.CharField(
                default="/home/studium/Masterarbeit/code/NASO/neural_architecture/datasets",
                max_length=100,
                null=True,
            ),
        ),
    ]
