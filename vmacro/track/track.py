from multiprocessing import Manager, Queue

import numpy as np

from trackers.djocsort.ocsort import TrackingConfig
from vmacro.config import TrackConfig
from vmacro.note import NoteClass
from vmacro.track.control import ControlWorker
from vmacro.track.tracking import TrackingWorker


class Track:
    def __init__(self, config: TrackConfig, tracking_config: TrackingConfig):
        from time import perf_counter
        self._key = config.key
        self._bbox = config.bbox
        self._note_classes = config.note_classes

        self._note_lifetime = config.note_lifetime
        self._default_speed = self._bbox[3] / self._note_lifetime

        self._tracking_config = tracking_config

        t0 = perf_counter()
        manager = Manager()
        self._cancelled = manager.Event()

        self._dets_queue: Queue = manager.Queue()
        self._trks_queue: Queue = manager.Queue()
        self._schedule_queue: Queue = manager.Queue()
        print('Create workers', perf_counter() - t0)

        self._tracking_worker = TrackingWorker(
            note_speed=self._default_speed,
            tracking_config=tracking_config,
            dets_queue=self._dets_queue,
            trks_queue=self._trks_queue,
            schedule_queue=self._schedule_queue,
            cancelled=self._cancelled,
        )
        #
        # self._control_worker = ControlWorker(
        #     key=self._key,
        #     note_speed=self._default_speed,
        #     bbox=self._bbox,
        #     schedule_queue=self._schedule_queue,
        #     cancelled=self._cancelled,
        # )

    def start(self):
        pass
        # self._tracking_worker.start()
        # self._control_worker.start()

    def stop(self, timeout=None):
        self._cancelled.set()
        self._tracking_worker.join(timeout)
        self._control_worker.join(timeout)

    def update_tracker(self, dets, img, timestamp):
        self._dets_queue.put((dets, img, timestamp))

    def get_tracking_results(self):
        return self._trks_queue.get()

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
