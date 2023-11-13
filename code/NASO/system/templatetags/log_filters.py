import re

from django import template

register = template.Library()


@register.filter
def colorize_log(log_content):
    # Split the log content into lines and reverse the order
    lines = log_content.splitlines()
    reversed_lines = reversed(lines)

    # Define regex patterns to match log levels, dates, and apply corresponding CSS classes
    log_level_patterns = {
        r"\bINFO\b": ("has-background-info", "INFO"),
        r"\bWARNING\b": ("has-background-warning", "WARNING"),
        r"\bERROR\b": ("has-background-danger", "ERROR"),
        r"\bCRITICAL\b": ("has-background-warning-dark", "CRITICAL"),
        r"\bSUCCESS\b": ("has-background-success", "SUCCESS"),
    }

    date_pattern = (
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3})"  # Modify the pattern as needed
    )

    # Apply CSS classes and log levels to each line
    colored_lines = []
    for line in reversed_lines:
        # Apply CSS class for the date portion (e.g., "2023-09-25 14:30:00")
        line = re.sub(date_pattern, r'<span class="text-green">\1</span>', line)

        # Apply CSS class for log levels
        for pattern, (css_class, log_level) in log_level_patterns.items():
            line = re.sub(
                pattern,
                f'<span class="{css_class} has-text-weight-bold text-uppercase">{log_level}</span>',
                line,
            )

        colored_lines.append(line)

    # Join the lines back together
    return "<br>".join(colored_lines)
