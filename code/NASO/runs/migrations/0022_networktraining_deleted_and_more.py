# Generated by Django 4.2.8 on 2023-12-17 23:08

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("runs", "0021_networktraining_description_networktraining_rate"),
    ]

    operations = [
        migrations.AddField(
            model_name="networktraining",
            name="deleted",
            field=models.DateTimeField(db_index=True, editable=False, null=True),
        ),
        migrations.AddField(
            model_name="networktraining",
            name="deleted_by_cascade",
            field=models.BooleanField(default=False, editable=False),
        ),
        migrations.AlterField(
            model_name="networktraining",
            name="naso_app_version",
            field=models.CharField(default="0.4.5", max_length=10),
        ),
    ]