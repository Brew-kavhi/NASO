import subprocess


def get_current_git_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except Exception:
        return None
