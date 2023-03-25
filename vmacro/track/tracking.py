from collections import Counter
from dataclasses import dataclass
from multiprocessing import Process, Queue, Event
from queue import Empty

import numpy as np

from vmacro.track.association import associate


@dataclass
class ObservationItem:
    track_id: int
    class_ids: list[int]
    first_y2: float
    last_y2: float
    first_timestamp: float
    last_timestamp: float


class TrackingWorker(Process):
    def __init__(
        self,
        bbox,
        note_speed,
        dets_queue: Queue,
        trks_queue: Queue,
        schedule_queue: Queue,
        cancelled: Event,
        *,
        iou_threshold: float = 0.3,
    ):
        super().__init__()
        self.daemon = True

        self._bbox = bbox
        self._note_speed = note_speed

        self._cancelled = cancelled
        self._dets_queue = dets_queue
        self._trks_queue = trks_queue
        self._schedule_queue = schedule_queue

        self._iou_threshold = iou_threshold

        self._observations: dict[int, ObservationItem] = {}

        # [x1, y1, x2, y2, det_conf, class_id, track_id, timestamp]
        self._trackings = np.empty((0, 8))
        self._max_id = -1

    @property
    def note_lifetime(self):
        return self._bbox[3] / self._note_speed

    def run(self):
        while not self._cancelled.is_set():
            try:
                dets, timestamp = self._dets_queue.get(timeout=1)
                self._update(dets, timestamp)
                self._trks_queue.put(self._trackings)
                self._schedule_queue.put((self._trackings, timestamp))
                self._post_update(timestamp)
            except Empty:
                ...

    def _update(self, dets: np.ndarray, timestamp: float):
        # det: [x1, y1, x2, y2, conf, class_id]
        # trk: [x1, y1, x2, y2, det_conf, class_id, track_id, timestamp]
        detections = dets[dets[:, 3].argsort()][::-1]  # sorted by bottom y
        track_ids = np.zeros(detections.shape[0], dtype=int)
        if self._trackings:
            skip, predictions = self._predict(timestamp)

            matched, unmatched_dets, unmatched_trks = associate(
                detections,
                predictions,
                self._iou_threshold,
            )

            for m in matched:
                det_index, trk_index = m
                track_id = self._trackings[skip + trk_index, 6]
                det = detections[det_index]
                track_ids[det_index] = track_id
                self._observations[track_id].class_ids.append(det[5])
                self._observations[track_id].last_y2 = det[3]
                self._observations[track_id].last_timestamp = timestamp
                det[5] = Counter(self._observations[track_id].class_ids).most_common(1)[0][0]
        else:
            unmatched_dets = np.arange(detections.shape[0], dtype=int)
            unmatched_trks = np.empty((0, 8))

        for det_index in unmatched_dets:
            self._max_id += 1
            track_ids[det_index] = self._max_id
            det = detections[det_index]
            self._observations[track_ids[det_index]] = ObservationItem(
                track_id=track_ids[det_index],
                class_ids=[det[5]],
                first_y2=det[3],
                first_timestamp=timestamp,
                last_y2=det[3],
                last_timestamp=timestamp,
            )

        unique_item_num = detections.shape[0] + unmatched_trks.shape[0]
        new_trackings = np.empty((unique_item_num, 8))

        new_trackings[:detections.shape[0], :7] = np.concatenate(
            (detections, track_ids[:, np.newaxis]),
            axis=1
        )
        new_trackings[:detections.shape[0], 7] = timestamp

        new_index = detections.shape[0]
        for index in unmatched_trks:
            new_trackings[new_index] = self._trackings[index]

        # sort the new trackings and save to instance
        self._trackings = new_trackings[new_trackings[:, 3].argsort()][::-1]

    def _predict(self, timestamp):
        results = self._trackings.copy()
        elapsed = timestamp - results[:, 7]
        dist = elapsed * self._note_speed
        results[:, [1, 3]] += dist
        for i, pred in enumerate(results):
            if pred[1] <= self._bbox[3]:
                # Find the first prediction that is still inside the game track
                return i, results[i:]
        return self._trackings.shape[0], np.empty((0, 8))

    def _post_update(self, timestamp):
        total_dist = 0
        total_time = 0
        to_del = []
        for observation in self._observations.values():
            if timestamp - observation.first_timestamp >= self.note_lifetime:
                to_del.append(observation.track_id)
            total_dist += observation.last_y2 - observation.first_y2
            total_time += observation.last_timestamp - observation.first_timestamp

        for track_id in to_del:
            self._observations.pop(track_id)

        if total_dist > 0 and total_time > 0:
            self._note_speed = (self._note_speed + total_dist / total_time) / 2
