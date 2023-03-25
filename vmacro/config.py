from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from vmacro.note import NoteClass

# network delay (ms) for executing key control commands
CONTROL_DELAY = 210  # 280

# delay (ms) for immediate releasing a key (key click)
CLICK_DELAY = 50 / 1e3

NOTE_CLASS_GROUP: dict[str, set[NoteClass]] = {
    'note': {'note', 'hold-start', 'hold-end'},
    'side': {'side-start', 'side-end'},
    'tb': {'tbnote', 'tbstart', 'tbend'},
    'extra': {'xnote', 'xstart', 'xend'},
}

KEY_NOTE_MAP: dict[str, set[NoteClass]] = {
    's': NOTE_CLASS_GROUP['note'],
    'd': NOTE_CLASS_GROUP['note'],
    'f': NOTE_CLASS_GROUP['note'],  # left-hand
    'j': NOTE_CLASS_GROUP['note'],
    'k': NOTE_CLASS_GROUP['note'],
    'l': NOTE_CLASS_GROUP['note'],  # right-hand
    'v': NOTE_CLASS_GROUP['extra'],
    'n': NOTE_CLASS_GROUP['extra'],  # extra red track for 8b l1, r1
    'e': NOTE_CLASS_GROUP['side'],
    'i': NOTE_CLASS_GROUP['side'],  # disk left and right (joystick down)
    # 'a': '', ';': '',  # fever, no detection needed, simply press regularly
}

FEVER_KEY = ';'

key_configs = {
    '4B': [
        ['s', 'd', 'k', 'l', ],
        ['e', 'i', ],
    ],
    '5B': [
        ['s', 'd', 'j', 'k', 'l', ],
        ['e', 'i', ],
    ],
    '6B': [
        ['s', 'd', 'f', 'j', 'k', 'l', ],
        ['e', 'i', ],
    ],
}

key_configs['4X'] = key_configs['4B'] + [['v', 'n']]

key_configs['5X'] = key_configs['5B'] + [['v', 'n']]

key_configs['8B'] = key_configs['6B'] + [['v', 'n']]

key_configs['XB'] = key_configs['8B'] + [['1', '2']]

BoardLocation = Literal['left', 'middle', 'right']

location_bbox: dict[BoardLocation, Tuple[float, float, float, float]] = {
    'left': (80.0, 0.0, 560.0, 745.0),
    # 'left': (102.0, 0.0, 494.0, 631.0),
    'middle': (0.0, 0.0, 0.0, 0.0),
    'right': (0.0, 0.0, 0.0, 0.0),
}


@dataclass
class TrackConfig:
    key: str
    bbox: np.ndarray
    note_classes: set[NoteClass]
    note_lifetime: float


class GameConfig:
    def __init__(self, mode, location, note_lifetime):
        self.mode = mode
        self.location = location
        self.bbox = location_bbox[location]

        mode_key_num = sum(len(keys) for keys in key_configs[mode])
        if isinstance(note_lifetime, list):
            if len(note_lifetime) == 1:
                self.note_lifetime = note_lifetime * mode_key_num
            elif len(note_lifetime) == mode_key_num:
                self.note_lifetime = note_lifetime
            else:
                raise ValueError(f"--note-lifetime must match the number of tracks of mode ‘{mode}’")
        else:
            self.note_lifetime = [note_lifetime] * mode_key_num

        self.track_configs = self._init_track_configs(mode, self.bbox[0], self.bbox[2])
        self.note_lifetime = note_lifetime

    def _init_track_configs(self, mode, start, end):
        length = end - start
        key_config = key_configs[mode]

        tracks = []
        j = 0
        for keys in key_config:
            track_start = start
            track_len = length / len(keys)
            for i, key in enumerate(keys):
                track_end = min(track_start + track_len, end)
                tracks.append(TrackConfig(
                    key=key,
                    note_classes=KEY_NOTE_MAP[key],
                    bbox=np.array([track_start, self.bbox[1], track_end, self.bbox[3]]),
                    note_lifetime=self.note_lifetime[j],
                ))
                track_start = track_end
                j += 1

        return tracks


if __name__ == '__main__':
    # print(KEY_NOTE_MAP.keys())
    config = GameConfig(mode='4B', location='left')
    for t in config.track_configs:
        print(t.bbox)
