import time
from multiprocessing import Manager, Queue, Event

import numpy as np

from vmacro.config import TrackConfig
from vmacro.note import NoteClass
from vmacro.track.control import ControlWorker
from vmacro.track.tracking import TrackingWorker


class Track:
    def __init__(
        self,
        config: TrackConfig,
        manager: Manager,
        trks_queue: Queue,
        cancelled: Event,
        log_key: str,
    ):
        self._key = config.key
        self._bbox = config.bbox
        self._note_classes = config.note_classes

        self._note_lifetime = config.note_lifetime
        self._default_speed = self._bbox[3] / (self._note_lifetime / 1e3)

        self._dets_queue: Queue = manager.Queue()
        self._schedule_queue: Queue = manager.Queue()

        self._tracking_worker = TrackingWorker(
            key=self._key,
            bbox=self._bbox,
            note_speed=self._default_speed,
            dets_queue=self._dets_queue,
            trks_queue=trks_queue,
            schedule_queue=self._schedule_queue,
            cancelled=cancelled,
            log_key=log_key,
        )

        self._control_worker = ControlWorker(
            key=self._key,
            note_speed=self._default_speed,
            bbox=self._bbox,
            schedule_queue=self._schedule_queue,
            cancelled=cancelled,
            log_key=log_key,
        )

    def start(self):
        self._tracking_worker.start()
        # self._control_worker.start()

    def stop(self, timeout=5):
        for second in range(timeout):
            if self._tracking_worker.is_alive() or self._control_worker.is_alive():
                time.sleep(1)
            else:
                break
        else:
            # Force kill if not exited gracefully
            self._tracking_worker.kill()
            self._control_worker.kill()
            self._tracking_worker.join(timeout=1)
            self._control_worker.join(timeout=1)

    def update_tracker(self, dets: np.ndarray, timestamp: float):
        self._dets_queue.put((dets, timestamp))

    def schedule(self, trks, timestamp):
        self._schedule_queue.put((trks, timestamp))

    def localize(self, bbox: np.ndarray, cls: NoteClass):
        nx1, _, nx2, _ = bbox
        tx1, _, tx2, _ = self._bbox
        if cls not in self._note_classes:
            return 0
        if tx1 <= nx2 and nx1 <= tx2:
            if nx1 < tx1:
                score = nx2 - tx1
            else:
                score = tx2 - nx1
            return score
        else:
            return 0

    def __hash__(self):
        return self._key.__hash__()

    def __eq__(self, other):
        return self._key == getattr(other, '_key', None)
