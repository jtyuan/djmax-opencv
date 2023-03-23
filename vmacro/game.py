import time

import keyboard
import win32api
import win32con
import win32gui
from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from vmacro.config import GameConfig, CLICK_DELAY, FEVER_KEY
from vmacro.note import Note
from vmacro.track import Track


class Game:
    def __init__(self, config: GameConfig, class_names):
        self._config = config
        self._class_names = class_names

        self._notes_history = {}
        self._scheduled_notes = set()

        self._target_hwnd = win32gui.FindWindow(None, "Chiaki | Stream")
        self._scheduler = BackgroundScheduler()

        self._min_hits = 1

        tracks = []
        for track_config in config.track_configs:
            tracks.append(Track(
                hwnd=self._target_hwnd,
                config=track_config,
                scheduler=self._scheduler,
            ))
        self._tracks = tracks

        self._scheduler.add_job(self._fever, 'interval', seconds=1)

        self._scheduler.start()

        self._last_scheduled = time.monotonic_ns()

    def _fever(self):
        # win32api.PostMessage(self._target_hwnd, win32con.WM_KEYDOWN, FEVER_KEY, 0)
        keyboard.press(FEVER_KEY)
        win32api.Sleep(CLICK_DELAY)
        # win32api.PostMessage(self._target_hwnd, win32con.WM_KEYUP, FEVER_KEY, 0)
        keyboard.release(FEVER_KEY)
        logger.debug("Fever!")

    def process(self, trks):
        for trk in trks:
            node_id = trk[4]
            # print("processing: {}", trk)
            if node_id not in self._scheduled_notes:
                self._notes_history.setdefault(node_id, [])
                bbox = trk[:4]
                node_cls = self._class_names[trk[5]]
                note = Note(
                    id=node_id,
                    bbox=bbox,
                    cls=node_cls,
                    timestamp=time.monotonic_ns(),
                )

                if len(self._notes_history[node_id]) > 0:
                    # Note did not move
                    logger.debug("{} {} {} {}", node_id, len(self._notes_history[node_id]),
                                 self._notes_history[node_id][-1].bbox, note.bbox)
                    if self._notes_history[node_id][-1].bbox[3] - note.bbox[3] <= 1e-6:
                        del self._notes_history[node_id][-1]

                self._notes_history[node_id].append(note)

                # print("note: {}; hist len: {}", note, len(self._notes_history[node_id]))
                if len(self._notes_history[node_id]) >= self._min_hits:
                    self._scheduled_notes.add(node_id)
                    track = self._assign_track(note)
                    if track:
                        first_note = self._notes_history[node_id][0]
                        speed = (note.bbox[3] - first_note.bbox[3]) / (note.timestamp - first_note.timestamp) * 1e6
                        logger.debug("observed speed: {}; pre-calculated speed:{}", speed, self._config.avg_speed)
                        track.schedule(note)
                    self._notes_history.pop(node_id)

    def process_dets(self, dets):
        now = time.monotonic_ns()
        if now - self._last_scheduled < 300 * 1e6:
            return
        self._last_scheduled = now
        for det in dets:
            conf = det[4]
            # print("processing: {}", det)1
            bbox = det[:4]
            node_cls = self._class_names[det[5]]
            note = Note(
                id=-1,
                bbox=bbox,
                cls=node_cls,
                timestamp=time.monotonic_ns(),
            )
            if note.bbox[3] >= self._config.bbox[3] * 0.8:
                continue

            # print("note: {}; hist len: {}", note, len(self._notes_history[node_id]))
            track = self._assign_track(note)
            if track:
                track.schedule(note)

    def _assign_track(self, note: Note):
        result = None
        max_score = 0
        for track in self._tracks:
            score = track.localize(note)
            if score > max_score:
                max_score = score
                result = track
        if result is None:
            logger.warning("Unable to assign track to note: {}", note)
        return result
