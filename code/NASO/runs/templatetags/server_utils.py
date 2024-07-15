from django import template
from django.utils.safestring import mark_safe

from neural_architecture.models.dataset import DatasetLoader

register = template.Library()


@register.simple_tag
def visualization_code(dataset_loader: DatasetLoader, predictions, targets):
    print(dataset_loader)
    if not dataset_loader.dataset_loader:
        dataset_loader.dataset_loader = dataset_loader.load_dataset_loader()
    return mark_safe(
        dataset_loader.dataset_loader.get_visualization_code(predictions, targets)
    )
