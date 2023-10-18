from django import template

register = template.Library()


@register.simple_tag
def get_attribute_tag(obj, attr_name):
    try:
        return obj.get(attr_name)
    except AttributeError:
        return None


@register.simple_tag
def get_metric(obj, attr_name):
    try:
        if isinstance(obj, list):
            obj = obj[0]
        return obj.get(attr_name)
    except AttributeError:
        return None
