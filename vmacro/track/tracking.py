import threading
import time
from collections import Counter
from dataclasses import dataclass
from queue import Empty, Queue

import numpy as np

from vmacro.config import special_key_type_map, KeyType
from vmacro.logger import init_logger
from vmacro.note import NoteClass
from vmacro.track.association import associate_v3, iou_batch_y
from vmacro.track.det_cleaner import DetCleaner


@dataclass
class ObservationItem:
    track_id: int
    class_ids: list[int]
    first_bbox: np.ndarray
    last_bbox: np.ndarray
    first_timestamp: float
    last_timestamp: float
    hit_streak: int


class TrackingWorker(threading.Thread):
    def __init__(
        self,
        key,
        bbox,
        note_speed,
        warmup_queue: Queue,
        dets_queue: Queue,
        tracking_output_queue: Queue,
        control_input_queue: Queue,
        trigger_queue: Queue,
        cancelled: threading.Event,
        log_key: str,
        class_names: dict[int, NoteClass],
        *,
        iou_threshold: float = 0.2,
    ):
        super().__init__()
        self.daemon = True

        self._log_key = log_key
        self._key = key
        self._bbox = bbox
        self._note_speed = note_speed
        self._class_names = class_names

        self._cancelled = cancelled
        self._warmup_queue = warmup_queue
        self._dets_queue = dets_queue
        self._tracking_output_queue = tracking_output_queue
        self._schedule_queue = control_input_queue
        self._trigger_queue = trigger_queue

        self._iou_threshold = iou_threshold

        self._observation_ttl = 0.1  # 0.1 s
        self._observations: dict[int, ObservationItem] = {}

        # [x1, y1, x2, y2, det_conf, class_id, track_id, timestamp, speed]
        self._dim_tracking = 9
        self._trackings = np.empty((0, self._dim_tracking))
        self._max_id = -1  # use float for consistency

        self._min_hits = 1
        self._track_bottom_tolerance = 100
        self._logger = None

        # Key type for special processing of side and extra notes
        self._key_type: KeyType = special_key_type_map.get(self._key, 'normal')

    def run(self):
        self._warmup_queue.put(True)  # inform the main process it's ready

        det_cleaner = DetCleaner(self._class_names)

        self._logger = init_logger(self._log_key, f"tracking-{self._key}")
        while not self._cancelled.is_set():
            try:
                dets, im0, timestamp = self._dets_queue.get(timeout=1)
                t0 = time.perf_counter()
                self._logger.debug(f"Input delay: {t0 - timestamp}")
                dets = det_cleaner.clean(self._key_type, dets, im0, self._logger)
                t1 = time.perf_counter()
                self._logger.debug(f"DetClean took: {t1 - t0}")
                output = self._update(dets, timestamp)
                self._post_update(timestamp)
                t0 = time.perf_counter()
                self._logger.debug(f"Update took: {t0 - t1}")
                self._tracking_output_queue.put(output)
                self._schedule_queue.put(output)
            except Empty:
                pass

    def _update(self, dets: np.ndarray, timestamp: float):
        # det: [x1, y1, x2, y2, conf, class_id]
        # trk: [x1, y1, x2, y2, det_conf, class_id, track_id, timestamp, speed]
        detections = dets[dets[:, 3].argsort()][::-1]  # sorted by bottom y
        self._logger.debug(f"input detections: {detections}")

        # Find and merge duplicated detections
        dets_iou = np.triu(iou_batch_y(detections, detections), 1)
        a = (dets_iou > 0.3).astype(np.int32)
        overlapped_indices = np.stack(np.where(a), axis=1)
        class_ids = [[] for _ in range(len(detections) - len(overlapped_indices))]
        rows_to_del = []
        for i, j in overlapped_indices:
            # Record ids of duplicated rows
            class_ids[i].append(detections[j, 5])
            rows_to_del.append(j)
        # Remove duplication
        detections = np.delete(detections, rows_to_del, axis=0)
        for i, det in enumerate(detections):
            # Record ids of unique rows
            class_ids[i].append(det[5])

        #       [ 0,  1,  2,  3,        4,        5,        6,         7,     8,           9]
        # pred: [x1, y1, x2, y2, det_conf, class_id, track_id, timestamp, speed, track_index]
        dropped_indices, predictions = self._predict(timestamp)
        if dropped_indices.any() > 0:
            self._logger.debug(f"Dropped {len(dropped_indices)} from trackings: "
                               f"{self._trackings[dropped_indices]}")

        matched, unmatched_dets, unmatched_preds = associate_v3(
            detections,
            predictions,
            dist_threshold=40,
            logger=self._logger
        )

        track_ids = np.zeros(len(detections), dtype=np.uint)
        for m in matched:
            det_index, pred_index = m
            track_id = int(predictions[pred_index, 6])
            det = detections[det_index]
            track_ids[det_index] = track_id
            self._observations[track_id].class_ids.extend(class_ids[det_index])
            self._observations[track_id].last_bbox = det[:4]
            self._observations[track_id].last_timestamp = timestamp
            self._observations[track_id].hit_streak += 1
            det[5] = Counter(self._observations[track_id].class_ids).most_common(1)[0][0]

        for det_index in unmatched_dets:
            self._max_id += 1
            track_id = track_ids[det_index] = int(self._max_id)
            det = detections[det_index]
            self._observations[track_id] = ObservationItem(
                track_id=int(track_ids[det_index]),
                class_ids=class_ids[det_index],
                first_bbox=det[:4],
                first_timestamp=timestamp,
                last_bbox=det[:4],
                last_timestamp=timestamp,
                hit_streak=0,
            )
            det[5] = Counter(self._observations[track_id].class_ids).most_common(1)[0][0]

        unique_item_num = len(detections) + len(unmatched_preds)
        new_trackings = np.empty((unique_item_num, self._dim_tracking), dtype=np.double)

        new_trackings[:len(detections), :7] = np.concatenate(
            (detections, track_ids[:, np.newaxis]),
            axis=1
        )

        new_index = len(detections)
        for pred_index in unmatched_preds:
            new_trackings[new_index] = self._trackings[int(predictions[pred_index, 9])]  # predictions[index, :9]  #
            self._observations[int(new_trackings[new_index][6])].hit_streak = 0
            new_index += 1

        new_trackings[:, 7] = timestamp
        new_trackings[:, 8] = self._note_speed

        # sort the new trackings and save to instance
        self._trackings = new_trackings[new_trackings[:, 3].argsort()][::-1]
        self._logger.debug(f"trackings {self._trackings}")
        output = []
        for trk in self._trackings:
            if self._observations[int(trk[6])].hit_streak >= self._min_hits:
                output.append(trk)
        return np.array(output, dtype=np.double)

    def _predict(self, timestamp) -> (np.ndarray, np.ndarray):
        # append a column of original index in the trackings array to be used after sort
        tracking_indices = np.arange(0, len(self._trackings), dtype=np.ushort)
        if self._trackings.any():
            results = np.column_stack([self._trackings, tracking_indices])
            elapsed = timestamp - results[:, 7, None]
            dist = elapsed * self._note_speed
            self._logger.debug(
                f"Time from last loop: {np.min(elapsed):.3f} s; "
                f"moved {np.min(dist):.1f} pixels"
            )
            results[:, [1, 3]] += dist
            results = results[results[:, 3].argsort()][::-1]  # sort by predicted y2 from large to small
            for i, pred in enumerate(results):
                if (
                    pred[1] <= self._bbox[3] + self._track_bottom_tolerance
                ):
                    # Find the first prediction that is still inside the game track
                    # and not marked as triggered by control
                    return results[:i, 9].astype(np.ushort), results[i:]
        return tracking_indices, np.empty((0, self._dim_tracking))

    def _post_update(self, timestamp):
        # total_dist = 0
        # total_time = 0
        to_del = self._observations.keys() - set(self._trackings[:, 6])
        # for observation in self._observations.keys():
        #     if timestamp - observation.last_timestamp > self._observation_ttl:
        #         self._logger.debug(
        #             f"Removing out-of-date tracking: {observation.track_id} "
        #             f"{observation.last_bbox}"
        #         )
        #         to_del.append(observation.track_id)
        #     # total_dist += observation.last_bbox[3] - observation.first_bbox[3]
        #     # total_time += observation.last_timestamp - observation.first_timestamp

        for track_id in to_del:
            self._observations.pop(track_id)

        # if total_dist > 0 and total_time > 0:
        #     self._note_speed = self._note_speed * 0.1 + (total_dist / total_time) * 0.9
        #     self._logger.debug(f"Track {self._key} speed updated to {self._note_speed}")
