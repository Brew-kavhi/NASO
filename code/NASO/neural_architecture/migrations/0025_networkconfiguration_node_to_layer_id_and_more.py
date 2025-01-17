# Generated by Django 4.2.6 on 2023-10-26 19:52

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("neural_architecture", "0024_autokerasmodel_callbacks"),
    ]

    operations = [
        migrations.AddField(
            model_name="networkconfiguration",
            name="node_to_layer_id",
            field=models.JSONField(default=dict),
        ),
        migrations.CreateModel(
            name="KerasNetworkTemplate",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=30, unique=True)),
                ("connections", models.JSONField(default=dict)),
                ("node_to_layer_id", models.JSONField(default=dict)),
                (
                    "layers",
                    models.ManyToManyField(
                        related_name="layers", to="neural_architecture.networklayer"
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="AutoKerasNetworkTemplate",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=30, unique=True)),
                ("connections", models.JSONField(default=dict)),
                ("node_to_layer_id", models.JSONField(default=dict)),
                (
                    "blocks",
                    models.ManyToManyField(
                        related_name="blocks", to="neural_architecture.autokerasnode"
                    ),
                ),
            ],
        ),
    ]
