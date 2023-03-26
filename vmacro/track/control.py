import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from enum import Enum
from multiprocessing import Process, Event, Queue, Lock
from queue import Empty
from typing import Optional

import keyboard
import numpy as np

from vmacro.config import CONTROL_DELAY, CLICK_DELAY
from vmacro.logger import init_logger
from vmacro.note import Note


class KeyAction(Enum):
    KEYDOWN = 0
    KEYUP = 1
    KEYPRESS = 2


class KeyState(Enum):
    DOWN = 0
    UP = 1


THROTTLE_DELAY_S = 55 / 1e3  # 30ms

BUCKET_SIZE = 200  # time bucket size. unit: 50; key actions are grouped into buckets by their planned time
MIN_HITS = 3  # at least MIN_HITS objects in bucket can trigger a schedule
MIN_DELAY = 300 / 1e3  # If lower than MIN_DELAY, schedule even when MIN_HITS is not reached

SCHEDULE_INTERVAL = 200 / 1e3

SCHEDULE_RANGE = [30, 700]
TRIGGER_RANGE = [550, 815]


class ControlWorker(Process):
    def __init__(
        self,
        key: str,
        note_speed: float,
        bbox: np.ndarray,
        schedule_queue: Queue,
        cancelled: Event,
        log_key: str,
    ):
        super().__init__()
        self.daemon = True

        self._log_key = log_key
        self._key = key
        self._bbox = bbox
        self._note_speed = note_speed
        self._status = KeyState.UP
        self._schedule_queue = schedule_queue
        self._cancelled = cancelled

        self._last_executed = 0
        self._last_note: Optional[Note] = None
        self._logger = None

    def run(self) -> None:
        self._logger = init_logger(self._log_key, f"control-{self._key}")
        self.keyup()  # reset key state
        key_check_lock = Lock()

        with ThreadPoolExecutor() as executor:
            while not self._cancelled.is_set():
                try:
                    trks, timestamp = self._schedule_queue.get(timeout=1)
                    for trk in trks:
                        executor.submit(self.schedule, trk, timestamp, key_check_lock)
                except Empty:
                    ...

    def schedule(self, trk: np.ndarray, timestamp: float, lock: Lock):
        # trk: [x1, y1, x2, y2, track_id, class_id, conf, queue]
        note = Note(
            bbox=trk[:4],
            id=trk[4],
            cls=trk[5],
            timestamp=timestamp,
            speed=self._note_speed,
        )
        delay = ((self._bbox[3] - note.bbox[3]) / note.speed - CONTROL_DELAY) / 1e3

        note_top = note.bbox[1]
        if SCHEDULE_RANGE[0] <= note_top <= SCHEDULE_RANGE[1]:
            detection_delay = time.perf_counter() - note.timestamp
            fixed_delay = max(delay - detection_delay, 0)
            target_time = (datetime.now() + timedelta(seconds=fixed_delay)).strftime("%Y-%m-%d %H:%M:%S.%f")
            self._logger.debug(
                "Schedule {}({})@{}:{:.3f} at speed {:.3f}pixel/ms at {} (schedule: {:.3f}ms) (delay: {:.3f}ms)",
                note.cls, self._key, note.bbox[3], note.timestamp, note.speed,
                target_time, fixed_delay * 1e3, detection_delay * 1e3
            )
            self._logger.debug(f"start waiting for {note.cls}({self._key})@{note.bbox[3]}:{note.timestamp:.3f} "
                         f"that is supposed to run at {target_time}")
            time.sleep(delay)
            self._logger.debug(f"done waiting for {note.cls}({self._key})@{note.bbox[3]}:{note.timestamp:.3f} at "
                         f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
            self._execute(note, lock)

    def keydown(self):
        keyboard.press(self._key)
        self._status = KeyState.DOWN
        self._logger.debug("KEYDOWN {}", self._key)

    def keyup(self):
        keyboard.release(self._key)
        self._status = KeyState.UP
        self._logger.debug("KEYUP {}", self._key)

    def keypress(self):
        keyboard.press(self._key)
        time.sleep(CLICK_DELAY)  # in thread, use normal sync sleep
        keyboard.release(self._key)
        self._status = KeyState.UP
        self._logger.debug("KEYPRESS {}", self._key)

    def _execute(self, note: Note, lock: Lock):
        now = time.perf_counter()  # s
        if note.cls in {'hold-start', 'side-start', 'tbstart', 'xstart'}:
            action = KeyAction.KEYDOWN
        elif note.cls in {'hold-end', 'side-end', 'tbend', 'xend'}:
            action = KeyAction.KEYUP
        else:
            action = KeyAction.KEYPRESS

        dist = (now - note.timestamp) * 1e3 * note.speed
        delta_h = 0  # (note.bbox[3] - note.bbox[1]) / 5
        cur_bbox = np.add(note.bbox, [0, dist - delta_h, 0, dist + delta_h])
        if not TRIGGER_RANGE[0] <= cur_bbox[3] <= TRIGGER_RANGE[1]:
            self._logger.debug("Dropped {}({})@{}:{:.3f} now at {:.3f}:{:.3f} which is out of range",
                         action, self._key, note.bbox[3], note.timestamp, cur_bbox[3], now)
            return
        with lock:
            # tx1 <= nx2 and nx1 <= tx2
            last_note = self._last_note
            if last_note:
                last_note_dist = (now - last_note.timestamp) * 1e3 * note.speed
                last_bbox = np.add(last_note.bbox, [0, last_note_dist, 0, last_note_dist])

                # self._logger.debug(f"last executed: {self._last_executed}; now: {now}")
                # self._logger.debug(f"last bb: {last_bbox}; final bb: {cur_bbox}")
                if last_bbox[1] <= cur_bbox[3] and cur_bbox[1] <= last_bbox[3]:
                    self._logger.debug("Dropped {} {} {} due to space adjacency with {}",
                                 action, self._key, note, last_note)
                    return
            if now - self._last_executed < THROTTLE_DELAY_S:
                self._logger.debug("Dropped {}({})@{}:{} due to time adjacency. Now: {}; Last execution: {}",
                             action, self._key, note.bbox[3], note.timestamp, now, self._last_executed)
                # Ignore consecutive notes that are likely to be the same note
                return
            self._last_note = Note(id=note.id, bbox=cur_bbox, cls=note.cls, timestamp=now, speed=note.speed)
            self._last_executed = now

        if self._status is KeyState.DOWN and action is not KeyAction.KEYDOWN:
            # If in down state, always release key whatever the action is (allows detection error)
            self.keyup()
        elif action is KeyAction.KEYDOWN:
            self.keydown()
        elif action is KeyAction.KEYPRESS:
            self.keypress()
        else:  # ignore Keyup when status is already up. Can be a previously failed keydown
            pass
