# Generated by Django 4.2.6 on 2023-10-19 09:05

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("neural_architecture", "0009_alter_autokerastunertype_module_name"),
    ]

    operations = [
        migrations.RenameField(
            model_name="autokerasnodetype",
            old_name="keras_type",
            new_name="autokeras_type",
        ),
    ]
