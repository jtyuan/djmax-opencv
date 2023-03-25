from multiprocessing import Manager, Queue

import numpy as np

from trackers.djocsort.ocsort import TrackingConfig
from vmacro.config import TrackConfig
from vmacro.note import NoteClass
from vmacro.track.control import ControlWorker
from vmacro.track.tracking import TrackingWorker


class Track:
    def __init__(self, config: TrackConfig, tracking_config: TrackingConfig, manager: Manager, embedder):
        from time import perf_counter
        self._key = config.key
        self._bbox = config.bbox
        self._note_classes = config.note_classes

        self._note_lifetime = config.note_lifetime
        self._default_speed = self._bbox[3] / self._note_lifetime

        self._tracking_config = tracking_config

        t0 = perf_counter()
        self._cancelled = manager.Event()
        self._schedule_queue: Queue = manager.Queue()

        self._tracking_worker = TrackingWorker(
            note_speed=self._default_speed,
            tracking_config=tracking_config,
            embedder=embedder,
        )

        self._control_worker = ControlWorker(
            key=self._key,
            note_speed=self._default_speed,
            bbox=self._bbox,
            schedule_queue=self._schedule_queue,
            cancelled=self._cancelled,
        )
        print('Create workers', perf_counter() - t0)

    def start(self):
        self._control_worker.start()

    def stop(self, timeout=None):
        self._cancelled.set()
        self._control_worker.join(timeout)

    def update_tracker(self, dets, img):
        return self._tracking_worker.run(dets, img)

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
