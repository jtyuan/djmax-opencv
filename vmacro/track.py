import time
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock

import keyboard
import win32api
import win32con
from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from vmacro.config import CLICK_DELAY, TrackConfig
from vmacro.note import Note


class KeyAction(Enum):
    KEYDOWN = 0
    KEYUP = 1
    KEYPRESS = 2


class KeyState(Enum):
    DOWN = 0
    UP = 1


THROTTLE_DELAY_NS = 20 * 1e6  # 20ms


class Track:
    def __init__(self, hwnd, config: TrackConfig, scheduler: BackgroundScheduler):
        self._hwnd = hwnd
        self._key = config.key
        self._bbox = config.bbox
        self._note_classes = config.note_classes
        self._last_executed = time.monotonic_ns()

        self._lock = Lock()
        self._status = KeyState.UP
        self._scheduler = scheduler

    def schedule(self, note: Note, speed: float):
        # action: KeyAction, delay: float
        if note.cls in {'hold-start', 'side-start', 'tbstart', 'xstart'}:
            action = KeyAction.KEYDOWN
        elif note.cls in {'hold-end', 'side-end', 'tbend', 'xend'}:
            action = KeyAction.KEYUP
        else:
            action = KeyAction.KEYPRESS
        delay = (self._bbox[3] - note.bbox[3]) / speed
        logger.debug(
            "Schedule note {} <{}> of speed {} to {} after {}",
            self._key, note, speed, action, delay,
        )
        self._scheduler.add_job(
            lambda: self._execute(action),
            'date',
            run_date=datetime.now() + timedelta(milliseconds=delay),
        )

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
        logger.debug("KEYDOWN {}", self._key)

    def keyup(self):
        # win32api.PostMessage(self._hwnd, win32con.WM_KEYUP, self._key, 0)
        keyboard.release(self._key)
        logger.debug("KEYUP {}", self._key)

    def keypress(self):
        self.keydown()
        win32api.Sleep(CLICK_DELAY)
        self.keyup()
        logger.debug("KEYPRESS {}", self._key)

    def _execute(self, action: KeyAction):
        with self._lock:
            now = time.monotonic_ns()
            if now - self._last_executed < THROTTLE_DELAY_NS:
                logger.debug("Dropped {} {}", action, self._key)
                # Ignore consecutive notes that are likely to come from detection error
                return
            self._last_executed = now

            if self._status is KeyState.DOWN:
                # If in down state, always release key whatever the action is (allows detection error)
                self.keyup()
                self._status = KeyState.UP
            elif action is KeyAction.KEYDOWN:
                self.keydown()
                self._status = KeyState.DOWN
            else:
                # state is UP and action in {KEYUP, KEYPRESS}, always press (KEYUP can be detection error)
                self.keypress()
                self._status = KeyState.UP
