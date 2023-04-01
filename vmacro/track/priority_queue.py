import heapq
import threading


class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()

    def put(self, item):
        with self.lock:
            heapq.heappush(self.queue, item)

    def get(self):
        with self.lock:
            return heapq.heappop(self.queue) if self.queue else None

    def replace(self, old, new):
        with self.lock:
            for i, item in enumerate(self.queue):
                if item == old:
                    self.queue[i] = new
                    heapq._siftup(self.queue, i)
                    heapq._siftdown(self.queue, 0, i)
                    break

    def peek(self):
        with self.lock:
            return self.queue[0] if self.queue else None

    def empty(self):
        with self.lock:
            return len(self.queue) == 0

    def __len__(self):
        with self.lock:
            return len(self.queue)


class KeyboardScheduler(threading.Thread):
    def __init__(self, cancelled: threading.Event):
        super().__init__()
        self.daemon = True
        self._event_queue = PriorityQueue()
        self._cancelled = cancelled

    def schedule(self, event_time, event_type, key):
        self._event_queue.put((event_time, event_type, key))
