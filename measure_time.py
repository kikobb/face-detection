import time


class MeasureTime:
    def __init__(self):
        self.timer = -1
        self.data = []
        self.fps = 0.0

    def start(self):
        self.timer = time.perf_counter()

    def stop(self):
        stop = time.perf_counter()
        if self.timer == -1:
            return
        duration = int(round((stop - self.timer) * 1000000))
        self.data.append(duration)
        self.fps = 1000000 / duration

    def print_data(self):
        print(';'.join(map(str, self.data)))

    def get_fps(self):
        if self.timer == -1:
            return None
        return self.fps

