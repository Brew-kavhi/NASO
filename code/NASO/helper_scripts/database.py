import time

import django


def lock_safe_db_operation(func, retries=3, delay=0.1):
    for i in range(retries):
        try:
            return func()
        except django.db.utils.OperationalError as e:
            if "database is locked" in str(e):
                if i == retries - 1:
                    raise
                time.sleep(delay)
            else:
                raise e
