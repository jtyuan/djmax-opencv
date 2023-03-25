from dataclasses import asdict
from multiprocessing import Process, Queue, Event
from queue import Empty

import numpy as np

from trackers.djocsort.ocsort import TrackingConfig
from trackers.multi_tracker_zoo import create_tracker


class TrackingWorker(Process):
    def __init__(
        self,
        note_speed,
        tracking_config: TrackingConfig,
        dets_queue: Queue,
        trks_queue: Queue,
        schedule_queue: Queue,
        cancelled: Event,
    ):
        super().__init__()
        self.daemon = True

        self._note_speed = note_speed
        self._tracking_config = tracking_config
        self._dets_queue = dets_queue
        self._trks_queue = trks_queue
        self._schedule_queue = schedule_queue
        self._cancelled = cancelled

    def run(self):
        tracker = create_tracker(
            **asdict(self._tracking_config),
            speed_prior=np.array([0, self._note_speed, 0, 0]),
        )
        if hasattr(tracker, 'model'):
            if hasattr(tracker.model, 'warmup'):
                tracker.model.warmup()

        while not self._cancelled.is_set():
            try:
                dets, img, timestamp = self._dets_queue.get(timeout=1)
                output = tracker.update(dets, img)
                self._trks_queue.put(output)
                self._schedule_queue.put((output, timestamp))
            except Empty:
                ...
