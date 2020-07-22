import time


class MeasureTime:
    def __init__(self):
        self.timer = -1
        self.data = []

    def start(self):
        self.timer = time.perf_counter()

    def stop(self):
        stop = time.perf_counter()
        if self.timer == -1:
            return
        self.data.append(int(round((stop - self.timer) * 1000000)))

    def print_data(self):
        print(';'.join(map(str, self.data)))
