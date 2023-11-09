import time


class Timer:
    def __init__(self):
        self.start_time = None
        self.total_time = 0

    def start(self):
        if self.start_time is None:
            self.start_time = time.time()

    def stop(self):
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.total_time += elapsed_time
            self.start_time = None
            return elapsed_time
        return 0

    def resume(self):
        if self.start_time is None:
            self.start_time = time.time()

    def get_total_time(self):
        return self.total_time
