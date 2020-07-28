import time

history_len = 20

class MeasureTime:
    def __init__(self):
        self.timer = -1
        self.data = []
        self.fps_history = [0.0] * history_len
        self._index = int(0)

    def start(self):
        self.timer = time.perf_counter()

    def stop(self):
        stop = time.perf_counter()
        if self.timer == -1:
            return
        duration = int(round((stop - self.timer) * 1000000))
        self.data.append(duration)
        self.fps_history[self._index] = round(1000000 / duration, 1)
        self._index += 1
        if self._index >= history_len:
            self._index = 0

    def print_data(self):
        print(';'.join(map(str, self.data)))

    def get_fps(self):
        if self.timer == -1:
            return None
        return self.fps_history[self._index]
    
    def get_extended_fps(self):
        if self.timer == -1:
            return None
        return self.fps_history[self._index], round(sum(self.fps_history) / len(self.fps_history), 1)
