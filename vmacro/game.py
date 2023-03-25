import warnings
from multiprocessing import Process, Queue
from queue import Empty

import keyboard
import trio
from apscheduler.schedulers.background import BackgroundScheduler

from vmacro.config import GameConfig, CLICK_DELAY, FEVER_KEY
from vmacro.logger import logger, init_logger
from vmacro.note import Note
from vmacro.track import Track

warnings.filterwarnings(
    "ignore",
    message="The localize method is no longer necessary, as this time zone supports the fold attribute",
)


class Game(Process):
    def __init__(self, config: GameConfig, class_names):
        super().__init__()
        self.daemon = True
        self.cancelled = False

        self._game_config = config
        self._class_names = class_names

        self._notes_history = {}
        self._scheduled_notes = set()

        self._scheduler = None
        self._min_hits = 1

        self._tracks = []
        self._task_queue = Queue()

        # trio.to_thread.run_sync()

    def run(self):
        trio.run(self._worker)

    async def _worker(self):
        init_logger()
        self._scheduler = BackgroundScheduler()  # executors={"default": ThreadPoolExecutor(96)})
        # self._scheduler.add_job(self._fever, 'interval', seconds=1)
        # self._scheduler.start()
        tracks = []
        for track_config in self._game_config.track_configs:
            tracks.append(Track(
                config=track_config,
                scheduler=self._scheduler,
            ))
        self._tracks = tracks

        async with trio.open_nursery() as nursery:
            while not self.cancelled:
                try:
                    dets, timestamp = await trio.to_thread.run_sync(self._get_data)
                    for det in dets:
                        nursery.start_soon(self._process_det, det, timestamp)
                except Empty:
                    ...

    def _get_data(self):
        return self._task_queue.get(timeout=1)

    @staticmethod
    def _fever():
        keyboard.press(FEVER_KEY)
        trio.sleep(CLICK_DELAY)
        keyboard.release(FEVER_KEY)

    def process(self, dets, timestamp):
        self._task_queue.put((dets, timestamp))

    async def _process_det(self, det, timestamp):
        conf = det[4]
        # print("processing: {}", det)1
        bbox = det[:4]
        node_cls = self._class_names[det[5]]
        note = Note(
            id=-1,
            bbox=bbox,
            cls=node_cls,
            timestamp=timestamp,
        )
        # if note.bbox[3] <= self._config.bbox[3] * 0.1:
        #     continue

        track: Track = self._assign_track(note)
        if track:
            await track.schedule(note)

    def _assign_track(self, note: Note):
        result = None
        max_score = 0
        for track in self._tracks:
            score = track.localize(note)
            if score > max_score:
                max_score = score
                result = track
        if result is None:
            logger.warning("Unable to assign track to note {}@[{}, {}]",
                           note.cls, note.bbox[2], note.bbox[3])
        return result
