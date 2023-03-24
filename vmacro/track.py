import time
from collections import Counter
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Optional

import keyboard
import numpy as np
import win32api
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


THROTTLE_DELAY_MS = 125  # 30ms

BUCKET_SIZE = 200  # time bucket size. unit: 50; key actions are grouped into buckets by their planned time
MIN_HITS = 3  # at least MIN_HITS objects in bucket can trigger a schedule
MIN_DELAY = 300 / 1e3  # If lower than MIN_DELAY, schedule even when MIN_HITS is not reached

SCHEDULE_INTERVAL = 200 / 1e3

SCHEDULE_RANGE = [30, 700]
TRIGGER_RANGE = [675, 815]


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

    def schedule(self, note: Note, timestamp: float):
        delay = max((self._bbox[3] - note.bbox[3]) / self._avg_speed - CONTROL_DELAY, 50)
        # target_time = _round(time.time_ns() / 1e6 + delay)

        note_top = note.bbox[1]
        if SCHEDULE_RANGE[0] <= note_top <= SCHEDULE_RANGE[1]:
            target_time = datetime.fromtimestamp((timestamp + delay / 1e3))

            self._scheduler.add_job(
                lambda: self._execute(note),
                'date',
                run_date=target_time,
            )
            logger.debug(
                "Schedule {}({}) at speed {:.3f}pixel/ms at {}",
                note.cls, self._key, self._avg_speed, target_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
            )
        # elif TRIGGER_RANGE[0] <= note_top <= TRIGGER_RANGE[1]:
        #     self._execute(action)
        # note.scheduled_ms = target_time
        # if target_time not in self._scheduled_time:
        #     self._cached_notes.setdefault(target_time, [])
        #     self._cached_notes[target_time].append(note)

    def _worker(self):
        now = time.time_ns()
        target_times = sorted(self._cached_notes.keys())
        schedule_times = filter(lambda t: t - now <= BUCKET_SIZE, target_times)

    def _schedule(self):
        """Schedule key actions for the cached notes in batch."""
        while True:
            with self._schedule_lock:
                target_times = sorted(self._cached_notes.keys())
                self._scheduled_time.update(target_times)
                if self._cached_notes:
                    logger.debug(f"cached notes of {self._key}: {self._cached_notes}")
                for target_time in target_times:
                    notes: list[Note] = self._cached_notes[target_time]
                    now = time.time()
                    target_time_s = target_time / 1e3
                    if len(notes) >= MIN_HITS or target_time_s - now <= MIN_DELAY:
                        self._cached_notes.pop(target_time)
                        cls = Counter(n.cls for n in notes).most_common(1)[0][0]
                        logger.debug(
                            f"most common class for {self._key} at {target_time}: {cls}; counter: {Counter(n.cls for n in notes)}")
                        if cls in {'hold-start', 'side-start', 'tbstart', 'xstart'}:
                            action = KeyAction.KEYDOWN
                        elif cls in {'hold-end', 'side-end', 'tbend', 'xend'}:
                            action = KeyAction.KEYUP
                        else:
                            action = KeyAction.KEYPRESS

                        if now <= target_time_s:
                            self._scheduler.add_job(
                                lambda: self._execute(action),
                                'date',
                                run_date=datetime.fromtimestamp(target_time_s),
                            )
                            logger.debug(
                                "Schedule {}({}) at speed {:.3f}pixel/ms at {:.3f}",
                                action, self._key, self._avg_speed, target_time,
                            )
                        else:
                            logger.warning("{} {} schedule expired", action, self._key)

            time.sleep(SCHEDULE_INTERVAL)

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
        # win32api.PostMessage(self._hwnd, win32con.WM_KEYDOWN, self._key, 0)
        keyboard.press(self._key)
        self._status = KeyState.DOWN
        logger.debug("KEYDOWN {}", self._key)

    def keyup(self):
        # win32api.PostMessage(self._hwnd, win32con.WM_KEYUP, self._key, 0)
        keyboard.release(self._key)
        self._status = KeyState.UP
        logger.debug("KEYUP {}", self._key)

    def keypress(self):
        keyboard.press(self._key)
        win32api.Sleep(CLICK_DELAY)
        keyboard.release(self._key)
        self._status = KeyState.UP
        logger.debug("KEYPRESS {}", self._key)

    def _execute(self, note: Note):
        now = time.perf_counter()  # s
        if note.cls in {'hold-start', 'side-start', 'tbstart', 'xstart'}:
            action = KeyAction.KEYDOWN
        elif note.cls in {'hold-end', 'side-end', 'tbend', 'xend'}:
            action = KeyAction.KEYUP
        else:
            action = KeyAction.KEYPRESS

        dist = (now - note.timestamp) * 1e3 * self._avg_speed
        delta_h = 0  # (note.bbox[3] - note.bbox[1]) / 5
        cur_bbox = np.add(note.bbox, [0, dist - delta_h, 0, dist + delta_h])
        if not TRIGGER_RANGE[0] <= cur_bbox[3] <= TRIGGER_RANGE[1]:
            logger.debug("Dropped {} {} {} (at {}) which is out of range",
                         action, self._key, note, cur_bbox[3])
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
            if now - self._last_executed < THROTTLE_DELAY_MS:
                logger.debug("Dropped {} {} {} due to time adjacency. Last execution: {}",
                             action, self._key, note,
                             datetime.fromtimestamp(self._last_executed / 1e3).strftime("%Y-%m-%d %H:%M:%S.%f"))
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
