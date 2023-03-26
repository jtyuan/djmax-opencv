from collections import Counter
from dataclasses import dataclass
from multiprocessing import Process, Queue, Event
from queue import Empty

import numpy as np

from vmacro.logger import init_logger
from vmacro.track.association import associate


@dataclass
class ObservationItem:
    track_id: int
    class_ids: list[int]
    first_bbox: np.ndarray
    last_bbox: np.ndarray
    first_timestamp: float
    last_timestamp: float
    hit_streak: int


class TrackingWorker(Process):
    def __init__(
        self,
        key,
        bbox,
        note_speed,
        dets_queue: Queue,
        trks_queue: Queue,
        schedule_queue: Queue,
        cancelled: Event,
        log_key: str,
        *,
        iou_threshold: float = 0.3,
    ):
        super().__init__()
        self.daemon = True

        self._log_key = log_key
        self._key = key
        self._bbox = bbox
        self._note_speed = note_speed

        self._cancelled = cancelled
        self._dets_queue = dets_queue
        self._trks_queue = trks_queue
        self._schedule_queue = schedule_queue

        self._iou_threshold = iou_threshold

        self._observations: dict[int, ObservationItem] = {}

        # [x1, y1, x2, y2, det_conf, class_id, track_id, timestamp]
        self._dim_tracking = 9
        self._trackings = np.empty((0, self._dim_tracking))
        self._max_id = -1.0  # use float for consistency

        self._min_hits = 3
        self._track_bottom_tolerance = 50
        self._logger = None

    def run(self):
        self._logger = init_logger(self._log_key, f"tracking-{self._key}")
        self._trks_queue.put(True)  # inform the main process it's ready
        while not self._cancelled.is_set():
            try:
                dets, timestamp = self._dets_queue.get(timeout=1)
                output = self._update(dets, timestamp)
                self._trks_queue.put(output)
                self._schedule_queue.put((output, timestamp))
                self._post_update()
            except Empty:
                ...

    def _update(self, dets: np.ndarray, timestamp: float):
        # det: [x1, y1, x2, y2, conf, class_id]
        # trk: [x1, y1, x2, y2, det_conf, class_id, track_id, timestamp]
        detections = dets[dets[:, 3].argsort()][::-1]  # sorted by bottom y
        track_ids = np.zeros(detections.shape[0], dtype=int)
        self._logger.debug(f"input detections: {detections}")
        if self._trackings.any():
            skip, predictions = self._predict(timestamp)

            matched, unmatched_dets, unmatched_trks = associate(
                detections,
                predictions,
                self._iou_threshold,
                logger=self._logger
            )

            for m in matched:
                det_index, trk_index = m
                track_id = self._trackings[skip + trk_index, 6]
                det = detections[det_index]
                track_ids[det_index] = track_id
                self._observations[track_id].class_ids.append(det[5])
                self._observations[track_id].last_bbox = det[:4]
                self._observations[track_id].last_timestamp = timestamp
                self._observations[track_id].hit_streak += 1
                det[5] = Counter(self._observations[track_id].class_ids).most_common(1)[0][0]
        else:
            unmatched_dets = np.arange(detections.shape[0], dtype=int)
            unmatched_trks = np.empty((0, self._dim_tracking))

        for det_index in unmatched_dets:
            self._max_id += 1
            track_ids[det_index] = self._max_id
            det = detections[det_index]
            self._observations[track_ids[det_index]] = ObservationItem(
                track_id=track_ids[det_index],
                class_ids=[det[5]],
                first_bbox=det[:4],
                first_timestamp=timestamp,
                last_bbox=det[:4],
                last_timestamp=timestamp,
                hit_streak=0,
            )

        unique_item_num = detections.shape[0] + unmatched_trks.shape[0]
        new_trackings = np.empty((unique_item_num, self._dim_tracking))

        new_trackings[:detections.shape[0], :7] = np.concatenate(
            (detections, track_ids[:, np.newaxis]),
            axis=1
        )
        new_trackings[:detections.shape[0], 7] = timestamp
        new_trackings[:, 8] = self._note_speed

        new_index = detections.shape[0]
        for index in unmatched_trks:
            new_trackings[new_index] = self._trackings[index]
            self._logger.debug(f'in loop {index} {self._trackings[index]}')
            self._observations[self._trackings[index][6]].hit_streak = 0
            new_index += 1

        # sort the new trackings and save to instance
        self._trackings = new_trackings[new_trackings[:, 3].argsort()][::-1]
        self._logger.debug(f"trackings {self._trackings}")
        output = []
        for trk in self._trackings:
            if self._observations[trk[6]].hit_streak >= self._min_hits:
                output.append(trk)
        return np.array(output)

    def _predict(self, timestamp):
        results = self._trackings.copy()
        elapsed = timestamp - results[:, 7, None]
        dist = elapsed * self._note_speed
        results[:, [1, 3]] += dist
        for i, pred in enumerate(results):
            if pred[1] <= self._bbox[3] + self._track_bottom_tolerance:
                # Find the first prediction that is still inside the game track
                return i, results[i:]
        return self._trackings.shape[0], np.empty((0, self._dim_tracking))

    def _post_update(self):
        total_dist = 0
        total_time = 0
        to_del = []
        for observation in self._observations.values():
            if observation.last_bbox[1] > self._bbox[3] + self._track_bottom_tolerance:
                to_del.append(observation.track_id)
            total_dist += observation.last_bbox[3] - observation.first_bbox[3]
            total_time += observation.last_timestamp - observation.first_timestamp

        for track_id in to_del:
            self._observations.pop(track_id)

        if total_dist > 0 and total_time > 0:
            self._note_speed = self._note_speed * 0.1 + (total_dist / total_time) * 0.9
            self._logger.debug(f"Track {self._key} speed updated to {self._note_speed}")
