from vmacro.control.config import TrackConfig


class Monitor:
    def __init__(self, track_config: TrackConfig):
        ...
        self._track = track_config
        self._key = ...

    def process(self, frame):
        ...
        # if not matched or matched.y > last_matched.y:
        #   if key hold: release
        #   else: key click
        # finally: record matched -> last_matched
