import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager

import keyboard
import numpy as np

from trackers.djocsort.ocsort import TrackingConfig
from trackers.djocsort.reid_multibackend import ReIDDetectMultiBackend
from vmacro.config import GameConfig, CLICK_DELAY, FEVER_KEY
from vmacro.logger import logger
from vmacro.note import NoteClass
from vmacro.track.track import Track

warnings.filterwarnings(
    "ignore",
    message="The localize method is no longer necessary, as this time zone supports the fold attribute",
)


class Game:
    def __init__(self, config: GameConfig, class_names, tracking_config: TrackingConfig):
        self._game_config = config
        self._class_names = class_names

        tracks = []

        manager = Manager()
        reid_weights = tracking_config.reid_weights
        device = tracking_config.device
        embedder = ReIDDetectMultiBackend(weights=reid_weights, device=device, fp16=False)
        for track_config in self._game_config.track_configs:
            tracks.append(Track(
                config=track_config,  # music/note track
                tracking_config=tracking_config,  # tracking not track
                manager=manager,
                embedder=embedder,
            ))
        self._tracks: list[Track] = tracks

        # fever_thread = threading.Thread(target=self._fever, daemon=True)
        # fever_thread.start()

    def start(self):
        for track in self._tracks:
            track.start()

    @staticmethod
    def _fever():
        while True:
            keyboard.press(FEVER_KEY)
            time.sleep(CLICK_DELAY)
            keyboard.release(FEVER_KEY)
            time.sleep(1)

    def process(self, detections, image, timestamp):
        # det: [x1, y1, x2, y2, conf, class_id]
        track_index = {}
        numpy_dets = detections.numpy()
        for i, det in enumerate(numpy_dets):
            bbox = det[:4]
            cls = self._class_names[det[5]]
            track: Track = self._assign_track(bbox, cls)
            track_index.setdefault(track, [])
            track_index[track].append(i)
        # with ThreadPoolExecutor(max_workers=len(track_index)) as executor:
        #     futures = {
        #         track: executor.submit(track.update_tracker, detections[index], image)
        #         for track, index in track_index.items()
        #     }
        # outputs = []
        # for track, future in futures.items():
        #     result = future.result()
        #     outputs.extend(result)
        #     track.schedule(result, timestamp)

        outputs = []
        for track, index in track_index.items():
            result = track.update_tracker(detections[index], image)
            outputs.extend(result)
            track.schedule(result, timestamp)

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
            logger.warning("Unable to assign track to note {}@[{}, {}]",
                           cls, bbox[2], bbox[3])
        return result
