from django.core.exceptions import ValidationError

from naso.constants import TENSORFLOW_DTYPES


def validate_dtype(value):
    if value not in TENSORFLOW_DTYPES:
        raise ValidationError(f"{value} is not a valid constant value.")
