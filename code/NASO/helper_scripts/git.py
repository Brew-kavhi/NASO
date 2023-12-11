import subprocess


def get_current_git_hash():
    """
    Returns the current git hash of the repository.

    Returns:
        str: The current git hash.
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except Exception:
        return None
