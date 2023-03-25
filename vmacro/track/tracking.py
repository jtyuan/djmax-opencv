from dataclasses import asdict

import numpy as np

from trackers.djocsort.ocsort import TrackingConfig
from trackers.multi_tracker_zoo import create_tracker


class TrackingWorker:
    def __init__(
        self,
        note_speed,
        tracking_config: TrackingConfig,
        embedder,
    ):
        super().__init__()
        self.daemon = True

        self._note_speed = note_speed
        self._tracking_config = tracking_config

        tracker = create_tracker(
            **asdict(self._tracking_config),
            speed_prior=np.array([0, self._note_speed, 0, 0]),
            embedder=embedder,
        )
        if hasattr(tracker, 'model'):
            if hasattr(tracker.model, 'warmup'):
                tracker.model.warmup()

        self._tracker = tracker

    def run(self, dets, img):
        return self._tracker.update(dets, img)
