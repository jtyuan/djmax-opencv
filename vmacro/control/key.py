import queue
import threading
from enum import Enum

import win32api
import win32con

from vmacro.control.config import DEFAULT_DELAY, CLICK_DELAY


class KeyAction(Enum):
    KEYDOWN = 0
    KEYUP = 1


class KeyState(Enum):
    DOWN = 0
    UP = 1


class Key:
    def __init__(self, hwnd, key: int):
        self._hwnd = hwnd
        self._key = key

        self._q = queue.Queue()
        self._status = KeyState.UP
        self._thread = threading.Thread(target=self._execute, daemon=True)

    def keydown(self, delay=DEFAULT_DELAY):
        self._q.put((KeyAction.KEYDOWN, delay))

    def keyup(self, delay=DEFAULT_DELAY):
        self._q.put((KeyAction.KEYUP, delay))

    def keypress(self, delay=DEFAULT_DELAY):
        self._q.put((KeyAction.KEYDOWN, delay))
        self._q.put((KeyAction.KEYUP, CLICK_DELAY))

    def _execute(self):
        while True:
            action, delay = self._q.get()
            win32api.Sleep(delay)
            if action is KeyAction.KEYDOWN:
                message_type = win32con.WM_KEYDOWN
                if self._status is KeyState.DOWN:
                    win32api.PostMessage(self._hwnd, win32con.WM_KEYUP, self._key, 0)
                self._status = KeyState.DOWN
            else:
                message_type = win32con.WM_KEYUP
                if self._status is KeyState.UP:
                    win32api.PostMessage(self._hwnd, win32con.WM_KEYDOWN, self._key, 0)
                    win32api.Sleep(CLICK_DELAY)
                self._status = KeyState.UP
            win32api.PostMessage(self._hwnd, message_type, self._key, 0)
