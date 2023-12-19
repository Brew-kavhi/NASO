# Generated by Django 4.2.8 on 2023-12-17 23:08

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            "neural_architecture",
            "0041_autokerasrun_description_autokerasrun_rate_and_more",
        ),
    ]

    operations = [
        migrations.AddField(
            model_name="autokerasrun",
            name="deleted",
            field=models.DateTimeField(db_index=True, editable=False, null=True),
        ),
        migrations.AddField(
            model_name="autokerasrun",
            name="deleted_by_cascade",
            field=models.BooleanField(default=False, editable=False),
        ),
        migrations.AddField(
            model_name="kerasmodelrun",
            name="deleted",
            field=models.DateTimeField(db_index=True, editable=False, null=True),
        ),
        migrations.AddField(
            model_name="kerasmodelrun",
            name="deleted_by_cascade",
            field=models.BooleanField(default=False, editable=False),
        ),
        migrations.AlterField(
            model_name="autokerasrun",
            name="naso_app_version",
            field=models.CharField(default="0.4.5", max_length=10),
        ),
        migrations.AlterField(
            model_name="kerasmodelrun",
            name="naso_app_version",
            field=models.CharField(default="0.4.5", max_length=10),
        ),
    ]
