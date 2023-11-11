# Generated by Django 4.2.7 on 2023-11-11 22:55

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0014_alter_networktraining_naso_app_version"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="evaluationparameters",
            name="callbacks",
        ),
        migrations.RemoveField(
            model_name="fitparameters",
            name="callbacks",
        ),
        migrations.AddField(
            model_name="evaluationparameters",
            name="callbacks",
            field=models.ManyToManyField(
                related_name="evaluation_callbacks", to="runs.callbackfunction"
            ),
        ),
        migrations.AddField(
            model_name="fitparameters",
            name="callbacks",
            field=models.ManyToManyField(
                related_name="fitparameters_callbacks", to="runs.callbackfunction"
            ),
        ),
    ]