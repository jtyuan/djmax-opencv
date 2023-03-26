import atexit
import time
import warnings
from datetime import datetime
from multiprocessing import Manager

import keyboard
import numpy as np

from vmacro.config import GameConfig, CLICK_DELAY, FEVER_KEY
from vmacro.logger import init_logger
from vmacro.note import NoteClass
from vmacro.track.track import Track

warnings.filterwarnings(
    "ignore",
    message="The localize method is no longer necessary, as this time zone supports the fold attribute",
)


class Game:
    def __init__(self, config: GameConfig, class_names):
        self._log_key = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        self._logger = init_logger(self._log_key)

        self._game_config = config
        self._class_names = class_names

        tracks = []

        manager = Manager()
        self._trks_queue = manager.Queue()
        self._cancelled = manager.Event()
        for track_config in self._game_config.track_configs:
            tracks.append(Track(
                config=track_config,  # music/note track
                manager=manager,
                trks_queue=self._trks_queue,
                cancelled=self._cancelled,
                log_key=self._log_key,
            ))
        self._tracks: list[Track] = tracks

        # fever_thread = threading.Thread(target=self._fever, daemon=True)
        # fever_thread.start()

    def start(self):
        atexit.register(self.stop)
        for track in self._tracks:
            track.start()
        for _ in self._tracks:
            # Wait for all tracks to be ready
            self._trks_queue.get()
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
            time.sleep(1)

    def process(self, detections, timestamp):
        # det: [x1, y1, x2, y2, conf, class_id]
        track_det_indices = {}
        numpy_dets = detections.numpy()
        for i, det in enumerate(numpy_dets):
            bbox = det[:4]
            cls = self._class_names[det[5]]
            track: Track = self._assign_track(bbox, cls)
            if track:
                track_det_indices.setdefault(track, [])
                track_det_indices[track].append(i)
        # with ThreadPoolExecutor(max_workers=len(track_det_indices)) as executor:
        #     futures = {
        #         track: executor.submit(track.update_tracker, detections[index], image)
        #         for track, index in track_det_indices.items()
        #     }
        # outputs = []
        # for track, future in futures.items():
        #     result = future.result()
        #     outputs.extend(result)
        #     track.schedule(result, timestamp)

        # outputs = []
        # for track, index in track_det_indices.items():
        #     result = track.update_tracker(detections[index], image)
        #     outputs.extend(result)
        #     track.schedule(result, timestamp)

        outputs = []
        for track, indices in track_det_indices.items():
            track.update_tracker(numpy_dets[indices], timestamp)
        for _ in track_det_indices:
            result = self._trks_queue.get()
            self._logger.debug(f"track result received: {result}")
            outputs.extend(result)
        return outputs

    def _assign_track(self, bbox: np.ndarray, cls: NoteClass):
        result = None
        max_score = 0
        for track in self._tracks:
            score = track.localize(bbox, cls)
            if score > max_score:
                max_score = score
                result = track
        if result is None:
            self._logger.warning("Unable to assign track to note {}@[{}, {}]",
                                 cls, bbox[2], bbox[3])
        return result
