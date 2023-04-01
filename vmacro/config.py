from dataclasses import dataclass
from typing import Literal

import numpy as np

from vmacro.note import NoteClass

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
    '1': NOTE_CLASS_GROUP['extra'],
    '2': NOTE_CLASS_GROUP['extra'],  # extra purple track for xb l2, r2
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

x_keys = ['v', 'n']
x2_keys = ['1', '2']

key_configs['4X'] = key_configs['4B'] + [x_keys]

key_configs['5X'] = key_configs['5B'] + [x_keys]

key_configs['8B'] = key_configs['6B'] + [x_keys]

key_configs['XB'] = key_configs['8B'] + [x2_keys]

BoardLocation = Literal['left', 'middle', 'right']

KeyType = Literal['side', 'x', 'x2', 'normal']

special_key_type_map: dict[str, KeyType] = {
    key_configs['XB'][1][0]: 'side',
    key_configs['XB'][1][1]: 'side',
    key_configs['XB'][2][0]: 'x',
    key_configs['XB'][2][1]: 'x',
    key_configs['XB'][3][0]: 'x2',
    key_configs['XB'][3][1]: 'x2',
}

frame_bbox: tuple[float, float, float, float] = (80.0, 0.0, 560.0, 745.0)


# 'left': (102.0, 0.0, 494.0, 631.0),


@dataclass
class TrackConfig:
    key: str
    bbox: np.ndarray
    note_classes: set[NoteClass]
    note_lifetime: float


class GameConfig:
    def __init__(self, mode, auto_fever: bool, note_lifetime):
        self.mode = mode
        self.bbox = frame_bbox

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
        self.auto_fever = auto_fever

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
    config = GameConfig(mode='4B')
    for t in config.track_configs:
        print(t.bbox)
