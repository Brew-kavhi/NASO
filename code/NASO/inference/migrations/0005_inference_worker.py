# Generated by Django 4.2.10 on 2024-03-13 00:20

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("inference", "0004_inference_flops"),
    ]

    operations = [
        migrations.AddField(
            model_name="inference",
            name="worker",
            field=models.CharField(default="pcsgs11", max_length=70),
            preserve_default=False,
        ),
    ]
