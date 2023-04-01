import atexit
import threading
import time
import warnings
from datetime import datetime
from queue import Queue

import numpy as np
from pynput.keyboard import Controller

from vmacro.config import GameConfig, FEVER_KEY
from vmacro.logger import init_logger
from vmacro.note import NoteClass
from vmacro.track.control import CLICK_DELAY
from vmacro.track.track import Track

warnings.filterwarnings(
    "ignore",
    message="The localize method is no longer necessary, as this time zone supports the fold attribute",
)

keyboard = Controller()


class Game:
    def __init__(self, config: GameConfig, class_names):
        self._log_key = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        self._logger = init_logger(self._log_key)

        self._game_config = config
        self._class_names = class_names

        tracks = []

        self._output_queue = Queue()
        self._cancelled = threading.Event()
        self._warmup_queue = Queue()
        self._loop_complete_queue = Queue()
        for track_config in self._game_config.track_configs:
            tracks.append(Track(
                config=track_config,  # music/note track
                warmup_queue=self._warmup_queue,
                tracking_output_queue=self._output_queue,
                loop_complete_queue=self._loop_complete_queue,
                cancelled=self._cancelled,
                log_key=self._log_key,
                class_names=class_names,
            ))
        self._tracks: list[Track] = tracks

        if config.auto_fever:
            fever_thread = threading.Thread(target=self._fever, daemon=True)
            fever_thread.start()

    def start(self):
        atexit.register(self.stop)
        for track in self._tracks:
            track.start()
        for _ in range(len(self._tracks) * 2):
            # Wait for all tracks to be ready
            self._warmup_queue.get()
        init_logger(self._log_key)

    def stop(self):
        self._cancelled.set()
        for track in self._tracks:
            track.stop(timeout=5)

    @staticmethod
    def _fever():
        while True:
            keyboard.press(FEVER_KEY)
            time.sleep(CLICK_DELAY)
            keyboard.release(FEVER_KEY)
            time.sleep(0.3)

    def process(self, detections, im0, timestamp):
        # det: [x1, y1, x2, y2, conf, class_id]
        t0 = time.perf_counter()
        track_det_indices = {}
        numpy_dets = detections.numpy()
        for i, det in enumerate(numpy_dets):
            bbox = det[:4]
            cls = self._class_names[det[5]]
            if det[1] == 0 or det[0] > self._game_config.bbox[2] or det[3] >= self._game_config.bbox[3]:
                continue
            track: Track = self._assign_track(bbox, cls, im0)
            if track:
                track_det_indices.setdefault(track, [])
                track_det_indices[track].append(i)

        outputs = []
        for track, indices in track_det_indices.items():
            track.update_tracker(numpy_dets[indices], im0, timestamp)
        for _ in track_det_indices:
            result = self._output_queue.get()
            outputs.extend(result)
        for _ in track_det_indices:
            self._loop_complete_queue.get()
        self._logger.debug(f"Game loop done in {time.perf_counter() - t0}s")
        return outputs

    def _assign_track(self, bbox: np.ndarray, cls: NoteClass, im0):
        result = None
        max_score = 0
        for track in self._tracks:
            score = track.localize(bbox, cls, im0)
            if score > max_score:
                max_score = score
                result = track
        # if result is None:
        #     self._logger.warning("Unable to assign track to note {}@[{}, {}]",
        #                          cls, bbox[2], bbox[3])
        return result
