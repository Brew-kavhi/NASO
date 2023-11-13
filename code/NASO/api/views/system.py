from django.http import JsonResponse
from system.templatetags.log_filters import colorize_log


def get_logs(request):
    log_file_path = "net.log"  # Adjust the path to your log file
    try:
        last_position = int(request.GET.get("last_position", 0))

        with open(log_file_path, "r", encoding="UTF-8") as log_file:
            log_file.seek(last_position)
            log_content = log_file.read()
            new_position = log_file.tell()
            log_content = colorize_log(log_content)

        return JsonResponse(
            {"log_content": log_content, "new_position": new_position}, safe=False
        )
    except FileNotFoundError:
        return JsonResponse({"log_content": "Log file not found."})
