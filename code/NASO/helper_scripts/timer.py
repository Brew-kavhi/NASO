import time


class Timer:
    """
    A simple timer class to measure elapsed time.
    """

    def __init__(self):
        """
        Constructor for the Timer class.
        """
        self.start_time = None
        self.total_time = 0

    def start(self):
        """
        Starts the timer.
        """
        if self.start_time is None:
            self.start_time = time.time()

    def stop(self):
        """
        Stops the timer and returns the elapsed time.
        """
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.total_time += elapsed_time
            self.start_time = None
            return elapsed_time
        return 0

    def resume(self):
        """
        Resumes the timer.
        """
        if self.start_time is None:
            self.start_time = time.time()

    def get_total_time(self):
        """
        Returns the total time elapsed.
        """
        return self.total_time
