import time
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Optional

import keyboard
import numpy as np
import trio
from apscheduler.schedulers.background import BackgroundScheduler

from vmacro.config import CLICK_DELAY, TrackConfig, CONTROL_DELAY
from vmacro.logger import logger
from vmacro.note import Note


class KeyAction(Enum):
    KEYDOWN = 0
    KEYUP = 1
    KEYPRESS = 2


class KeyState(Enum):
    DOWN = 0
    UP = 1


THROTTLE_DELAY_S = 125 / 1e3  # 30ms

BUCKET_SIZE = 200  # time bucket size. unit: 50; key actions are grouped into buckets by their planned time
MIN_HITS = 3  # at least MIN_HITS objects in bucket can trigger a schedule
MIN_DELAY = 300 / 1e3  # If lower than MIN_DELAY, schedule even when MIN_HITS is not reached

SCHEDULE_INTERVAL = 200 / 1e3

SCHEDULE_RANGE = [30, 700]
TRIGGER_RANGE = [550, 815]


def _round(delay):
    return int(BUCKET_SIZE * round(delay / BUCKET_SIZE))


class Track:
    def __init__(self, config: TrackConfig, scheduler: BackgroundScheduler):
        self._key = config.key
        self._bbox = config.bbox
        self._note_classes = config.note_classes
        self._last_executed = 0

        self._execute_lock = Lock()
        self._schedule_lock = Lock()
        self._status = KeyState.UP
        self._scheduler = scheduler

        self._avg_speed = config.avg_speed

        self._cached_notes = {}
        self._scheduled_time = set()
        self._last_note: Optional[Note] = None

        self.keyup()  # reset key state
        # self._thread = threading.Thread(target=self._schedule, daemon=True).start()

    async def schedule(self, note: Note):
        delay = ((self._bbox[3] - note.bbox[3]) / self._avg_speed - CONTROL_DELAY) / 1e3

        note_top = note.bbox[1]
        if SCHEDULE_RANGE[0] <= note_top <= SCHEDULE_RANGE[1]:
            detection_delay = time.perf_counter() - note.timestamp
            fixed_delay = max(delay - detection_delay, 0)
            target_time = (datetime.now() + timedelta(seconds=fixed_delay)).strftime("%Y-%m-%d %H:%M:%S.%f")
            logger.debug(
                "Schedule {}({})@{}:{:.3f} at speed {:.3f}pixel/ms at {} (schedule: {:.3f}ms) (delay: {:.3f}ms)",
                note.cls, self._key, note.bbox[3], note.timestamp, self._avg_speed,
                target_time, fixed_delay * 1e3, detection_delay * 1e3
            )
            await trio.to_thread.run_sync(self._execute, note, fixed_delay, target_time)

    def localize(self, note: Note):
        nx1, ny1, nx2, ny2 = note.bbox
        tx1, ty1, tx2, ty2 = self._bbox
        if note.cls not in self._note_classes:
            return 0
        if (nx2 - nx1) > (tx2 - tx1) * 1.5:
            # long note, shouldn't be in this track
            return 0
        if tx1 <= nx2 and nx1 <= tx2:
            if nx1 < tx1:
                score = nx2 - tx1
            else:
                score = tx2 - nx1
            return score
        else:
            return 0

    def keydown(self):
        keyboard.press(self._key)
        self._status = KeyState.DOWN
        logger.debug("KEYDOWN {}", self._key)

    def keyup(self):
        keyboard.release(self._key)
        self._status = KeyState.UP
        logger.debug("KEYUP {}", self._key)

    def keypress(self):
        keyboard.press(self._key)
        time.sleep(CLICK_DELAY)  # in thread, use normal sync sleep
        keyboard.release(self._key)
        self._status = KeyState.UP
        logger.debug("KEYPRESS {}", self._key)

    def _execute(self, note: Note, delay: float, target):
        logger.debug(f"start waiting for {note.cls}({self._key})@{note.bbox[3]}:{note.timestamp:.3f} "
                     f"that is supposed to run at {target}")
        time.sleep(delay)
        logger.debug(f"done waiting for {note.cls}({self._key})@{note.bbox[3]}:{note.timestamp:.3f} at "
                     f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
        now = time.perf_counter()  # s
        if note.cls in {'hold-start', 'side-start', 'tbstart', 'xstart'}:
            action = KeyAction.KEYDOWN
        elif note.cls in {'hold-end', 'side-end', 'tbend', 'xend'}:
            action = KeyAction.KEYUP
        else:
            action = KeyAction.KEYPRESS

        dist = (now - note.timestamp) * 1e3 * self._avg_speed
        delta_h = (note.bbox[3] - note.bbox[1]) / 5
        cur_bbox = np.add(note.bbox, [0, dist - delta_h, 0, dist + delta_h])
        if not TRIGGER_RANGE[0] <= cur_bbox[3] <= TRIGGER_RANGE[1]:
            logger.debug("Dropped {}({})@{}:{:.3f} now at {:.3f}:{:.3f} which is out of range",
                         action, self._key, note.bbox[3], note.timestamp, cur_bbox[3], now)
            return
        with self._execute_lock:
            # tx1 <= nx2 and nx1 <= tx2
            last_note = self._last_note
            if last_note:
                last_note_dist = (now - last_note.timestamp) * 1e3 * self._avg_speed
                last_bbox = np.add(last_note.bbox, [0, last_note_dist, 0, last_note_dist])

                # logger.debug(f"last executed: {self._last_executed}; now: {now}")
                # logger.debug(f"last bb: {last_bbox}; final bb: {cur_bbox}")
                if last_bbox[1] <= cur_bbox[3] and \
                        cur_bbox[1] <= last_bbox[3]:
                    logger.debug("Dropped {} {} {} due to space adjacency with {}",
                                 action, self._key, note, last_note)
                    return
            if now - self._last_executed < THROTTLE_DELAY_S:
                logger.debug("Dropped {}({})@{}:{} due to time adjacency. Now: {}; Last execution: {}",
                             action, self._key, note.bbox[3], note.timestamp, now, self._last_executed)
                # Ignore consecutive notes that are likely to be the same note
                return
            self._last_note = Note(id=note.id, bbox=cur_bbox, cls=note.cls, timestamp=now)
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
