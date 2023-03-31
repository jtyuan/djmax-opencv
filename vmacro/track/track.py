import time
from queue import Queue
from threading import Event

import numpy as np

from vmacro.config import TrackConfig, KeyType, special_key_type_map
from vmacro.note import NoteClass
from vmacro.track.control import ControlWorker
from vmacro.track.det_cleaner import DetCleaner
from vmacro.track.tracking import TrackingWorker


class Track:
    def __init__(
        self,
        config: TrackConfig,
        warmup_queue: Queue,
        trks_queue: Queue,
        loop_complete_queue: Queue,
        cancelled: Event,
        log_key: str,
        class_names: dict[int, NoteClass],
    ):
        self._key = config.key
        self._bbox = config.bbox
        self._note_classes = config.note_classes

        self._note_lifetime = config.note_lifetime
        self._default_speed = self._bbox[3] / (self._note_lifetime / 1e3)

        self._dets_queue: Queue = Queue()
        self._schedule_queue: Queue = Queue()
        self._scheduled_queue: Queue = Queue()
        self._trigger_queue: Queue = Queue()

        self._tracking_worker = TrackingWorker(
            key=self._key,
            bbox=self._bbox,
            note_speed=self._default_speed,
            warmup_queue=warmup_queue,
            dets_queue=self._dets_queue,
            trks_queue=trks_queue,
            schedule_queue=self._schedule_queue,
            trigger_queue=self._trigger_queue,
            cancelled=cancelled,
            log_key=log_key,
            class_names=class_names,
        )

        self._control_worker = ControlWorker(
            key=self._key,
            note_speed=self._default_speed,
            bbox=self._bbox,
            warmup_queue=warmup_queue,
            schedule_queue=self._schedule_queue,
            trigger_queue=self._trigger_queue,
            loop_complete_queue=loop_complete_queue,
            cancelled=cancelled,
            log_key=log_key,
            class_names=class_names,
        )

        self._det_cleaner = DetCleaner(class_names)
        self._key_type: KeyType = special_key_type_map.get(self._key, 'normal')

    def start(self):
        self._tracking_worker.start()
        self._control_worker.start()

    def stop(self, timeout=5):
        for second in range(timeout):
            if self._tracking_worker.is_alive() or self._control_worker.is_alive():
                time.sleep(1)
            else:
                break

    def update_tracker(self, dets: np.ndarray, im0, timestamp: float):
        self._dets_queue.put((dets, im0, timestamp))

    def localize(self, bbox: np.ndarray, cls: NoteClass, im0):
        # from loguru import logger
        if cls not in self._note_classes:
            return 0
        nx1, _, nx2, _ = bbox
        tx1, _, tx2, _ = self._bbox
        # Calculate score with 1D IoU
        x1 = max(nx1, tx1)
        x2 = min(nx2, tx2)
        overlap = max(0, x2 - x1)
        score = overlap / (nx2 - nx1 + tx2 - tx1 - overlap)

        # if self._key_type in {'x', 'x2'}:
        #     key_type = self._det_cleaner.get_key_type(bbox, im0)
        #     logger.info(f"KeyType: {self._key_type} vs {key_type}")
        #     score += 1 if key_type == self._key_type else -1

        return score

    def __hash__(self):
        return self._key.__hash__()

    def __eq__(self, other):
        return self._key == getattr(other, '_key', None)
