import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue
from threading import Thread, Event, Lock
from typing import Optional

import numpy as np
from pynput.keyboard import Controller

from vmacro.logger import init_logger
from vmacro.note import Note, NoteClass

# network delay (ms) for executing key control commands
CONTROL_DELAY = 110 / 1e3  # 280

# delay (ms) for immediate releasing a key (key click)
CLICK_DELAY = 80 / 1e3

# Wake the waiting thread in advance to handle special cases like unreleased keydown
CONTROL_PADDING = 50 / 1e3

THROTTLE_DELAY_S = 80 / 1e3  # ms
SCHEDULE_CHANGE_THRESHOLD = 300 / 1e3  # ms
SCHEDULE_DIFF_THRESHOLD = 50 / 1e3  # ms

SCHEDULE_RANGE = [30, 740]
TRIGGER_RANGE = [600, 815]

keyboard = Controller()


class KeyAction(Enum):
    KEYDOWN = 0
    KEYUP = 1
    KEYPRESS = 2


class KeyState(Enum):
    DOWN = 0
    UP = 1


@dataclass
class ControlSchedule:
    note: Note
    target_time: float
    cancelled: Event


class ControlWorker(Thread):
    def __init__(
        self,
        key: str,
        note_speed: float,
        bbox: np.ndarray,
        warmup_queue: Queue,
        schedule_queue: Queue,
        trigger_queue: Queue,
        loop_complete_queue: Queue,
        cancelled: Event,
        log_key: str,
        class_names: dict[int, NoteClass],
    ):
        super().__init__()
        self.daemon = True

        self._log_key = log_key
        self._key = key
        self._bbox = bbox
        self._note_speed = note_speed
        self._status = KeyState.UP
        self._warmup_queue = warmup_queue
        self._schedule_queue = schedule_queue
        self._trigger_queue = trigger_queue
        self._loop_complete_queue = loop_complete_queue
        self._cancelled = cancelled

        self._last_executed = 0
        self._last_note: Optional[Note] = None
        self._logger = None

        self._class_names = class_names
        self._futures = {}  # id -> schedule future
        self._schedules: dict[int, ControlSchedule] = {}  # id -> schedule

    def run(self) -> None:
        self._warmup_queue.put(True)
        self._logger = init_logger(self._log_key, f"control-{self._key}")
        self.keyup('init')  # reset key state
        key_lock = Lock()

        with ThreadPoolExecutor() as executor:
            while not self._cancelled.is_set():
                try:
                    trks = self._schedule_queue.get(timeout=1)
                except Empty:
                    trks = []
                try:
                    for trk in trks:
                        note = Note(
                            bbox=trk[:4],
                            id=int(trk[6]),
                            cls=self._class_names[trk[5]],
                            timestamp=trk[7],
                            speed=trk[8],
                        )
                        # trk: # [x1, y1, x2, y2, det_conf, class_id, track_id, timestamp, speed]
                        delay = (self._bbox[3] - note.bbox[3]) / note.speed - CONTROL_DELAY  # s

                        note_top = note.bbox[1]
                        if SCHEDULE_RANGE[0] <= note_top <= SCHEDULE_RANGE[1]:
                            now = time.perf_counter()
                            detection_delay = now - note.timestamp
                            fixed_delay = max(delay - detection_delay, 0)
                            target_time = now + fixed_delay

                            prev_future = self._futures.get(note.id)
                            if prev_future is not None:
                                schedule = self._schedules.get(note.id)

                                prev_note = schedule.note
                                prev_target = schedule.target_time
                                not_execute_soon = abs(prev_target - now) > SCHEDULE_CHANGE_THRESHOLD
                                class_changed = prev_note.cls != note.cls
                                target_time_changed = abs(prev_target - target_time) > SCHEDULE_DIFF_THRESHOLD
                                if (not_execute_soon and target_time_changed) or class_changed:
                                    schedule.cancelled.set()
                                    prev_future.cancel()
                                    self._logger.debug(
                                        "Cancelled previous schedule of {}#{}@{}:{:.3f}. "
                                        "target time: {:.3f} -> {:.3f}; class: {} -> {}",
                                        prev_note.cls, prev_note.id, prev_note.bbox[3], prev_note.timestamp,
                                        prev_target, target_time, prev_note.cls, note.cls,
                                    )
                                else:
                                    self._logger.debug(
                                        "Skipped duplicated {}#{}@{}:{:.3f}. "
                                        "target time: {:.3f} (current: {:.3f})",
                                        note.cls, note.id, note.bbox[3], note.timestamp,
                                        target_time, schedule.target_time
                                    )
                                    continue

                            self._schedules[note.id] = ControlSchedule(note, target_time, Event())

                            self._logger.debug(
                                "Schedule {}#{}@{:.1f}:{:.3f} at speed {:.0f}pixel/s to {:.3f} "
                                "(schedule: {:.0f}ms) (time loss: {:.0f}ms)",
                                note.cls, note.id, note.bbox[3], note.timestamp, note.speed,
                                target_time, fixed_delay * 1e3, detection_delay * 1e3
                            )
                            self._futures[note.id] = executor.submit(
                                self._execute_soon, fixed_delay, target_time, note, key_lock
                            )
                    self._loop_complete_queue.put(True)
                except:
                    self._logger.exception("Failed to schedule")
                    self._loop_complete_queue.put(False)

    def keydown(self, note_id):
        self._logger.debug(f"KEYDOWN {self._key} #{note_id} at {time.perf_counter():.3f}")
        keyboard.press(self._key)
        self._status = KeyState.DOWN
        self._logger.debug(f"KEYDOWN {self._key} #{note_id} executed")

    def keyup(self, note_id):
        self._logger.debug(f"KEYUP {self._key} #{note_id} at {time.perf_counter():.3f}")
        keyboard.release(self._key)
        self._status = KeyState.UP
        self._logger.debug(f"KEYUP {self._key} #{note_id} executed")

    def keypress(self, note_id):
        self._logger.debug(f"KEYPRESS {self._key} #{note_id} at {time.perf_counter():.3f}")
        keyboard.press(self._key)
        time.sleep(CLICK_DELAY)  # in thread, use normal sync sleep
        keyboard.release(self._key)
        self._status = KeyState.UP
        self._logger.debug(f"KEYPRESS {self._key} #{note_id} executed")

    def _execute_soon(self, delay: float, target_time: float, note: Note, lock: Lock):
        self._logger.debug(
            f"Start waiting for {note.cls}#{note.id}@{note.bbox[3]:.1f}"
            f":{note.timestamp:.3f} at {time.perf_counter():.3f} for {delay:3f}s"
        )
        time.sleep(delay - CONTROL_PADDING)
        now = time.perf_counter()
        self._logger.debug(
            f"Done waiting for {note.cls}#{note.id}@{note.bbox[3]:.1f}"
            f":{note.timestamp:.3f} at {now:.3f} (difference from schedule: {target_time - now:.3f})"
        )
        self._execute(note, target_time, lock)

    def _execute(self, note: Note, target_time: float, lock: Lock):
        schedule = self._schedules.get(note.id)
        if not schedule:
            self._logger.warning(f"Schedule of note {note.cls}#{note.id}@{note.bbox[3]:.1f} "
                                 f"run without schedule object.")
        if schedule.cancelled.is_set():
            self._logger.debug(f"Schedule of note {note.cls}#{note.id}@{note.bbox[3]:.1f} "
                               f"cancelled before execution.")
            return

        now = time.perf_counter()  # s
        if note.cls in {'hold-start', 'side-start', 'tbstart', 'xstart'}:
            action = KeyAction.KEYDOWN
        elif note.cls in {'hold-end', 'side-end', 'tbend', 'xend'}:
            action = KeyAction.KEYUP
        else:
            action = KeyAction.KEYPRESS

        dist = (now - note.timestamp) * note.speed
        delta_h = 0  # (note.bbox[3] - note.bbox[1]) / 5
        cur_bbox = np.add(note.bbox, [0, dist - delta_h, 0, dist + delta_h])
        # if not TRIGGER_RANGE[0] <= cur_bbox[3] <= TRIGGER_RANGE[1]:
        #     self._logger.debug("Dropped {}({})@{:.1f}:{:.3f} now at {:.1f}:{:.3f} which is out of range",
        #                        action, note.id, note.bbox[3], note.timestamp, cur_bbox[3], now)
        #     return
        with lock:
            # tx1 <= nx2 and nx1 <= tx2
            last_note = self._last_note
            if last_note:
                last_note_dist = (now - last_note.timestamp) * note.speed
                last_bbox = np.add(last_note.bbox, [0, last_note_dist, 0, last_note_dist])

                # self._logger.debug(f"last executed: {self._last_executed}; now: {now}")
                # self._logger.debug(f"last bb: {last_bbox}; final bb: {cur_bbox}")
                if last_bbox[1] <= cur_bbox[3] and cur_bbox[1] <= last_bbox[3]:
                    self._logger.debug("Dropped {}({})@{:.1f}:{:.3f} due to space adjacency with"
                                       "({})@{:.1f}:{:.3f}",
                                       action, note.id, note.bbox[3], note.timestamp,
                                       last_note.id, last_note.bbox[3], last_note.timestamp)
                    return
                if last_note.id == note.id and now - self._last_executed < THROTTLE_DELAY_S:
                    self._logger.debug("Dropped {}({})@{:.1f}:{:.3f} due to time adjacency."
                                       "Now: {:.3f}; Last execution: {:.3f}",
                                       action, note.id, note.bbox[3], note.timestamp, now, self._last_executed)
                    # Ignore consecutive notes that are likely to be the same note
                    return
            self._last_note = Note(id=note.id, bbox=cur_bbox, cls=note.cls, timestamp=now, speed=note.speed)
            self._last_executed = now

        if self._status is KeyState.DOWN:
            # If in down state, always release key whatever the action is (allows detection error)
            if action is KeyAction.KEYDOWN:
                if self._last_note.id != note.id:
                    # KEYUP for consecutive KEYDOWN if not duplicated KEYDOWN of the same node
                    self.keyup(note.id)
            else:
                # For KEYUP and KEYPRESS, just key up first
                self.keyup(note.id)

            # if action is not KeyAction.KEYUP:
            #     # Wait for a delay to prevent the following keydown/keypress from being discarded
        self._logger.debug("Sleep again")
        time.sleep(max(target_time - time.perf_counter(), 0))
        self._logger.debug("Wakeup again")

        if action is KeyAction.KEYDOWN:
            if self._status is KeyState.DOWN and self._last_note.id != note.id:
                self.keyup(note.id)
                time.sleep(CLICK_DELAY)
            self.keydown(note.id)
        elif action is KeyAction.KEYPRESS:
            self.keypress(note.id)
        else:  # ignore Keyup when status is already up. Can be a previously failed keydown
            pass

        self._trigger_queue.put((note.id, time.perf_counter()))
