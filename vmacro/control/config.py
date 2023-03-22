from dataclasses import dataclass
from typing import Literal, Tuple

# delay (ms) before executing key control commands
DEFAULT_DELAY = 0

# delay (ms) for immediate releasing a key (key click)
CLICK_DELAY = 100

start_y = 0
end_y = 1080

templates = {
    'extra_left': ['left8b.jpg', 'left8bHoldS.jpg', 'left8bHoldM.jpg', 'left8bHoldE.jpg'],
    'extra_right': ['right8b.jpg', 'right8bHoldS.jpg', 'right8bHoldM.jpg', 'right8bHoldE.jpg'],
    'side': ['', 'sideHoldS.jpg', 'sideHoldM.jpg', 'sideHoldE.jpg'],
    'key1': ['{theme}/whiteNote{mode}.jpg', '{theme}/whiteHoldS{mode}.jpg',
             '{theme}/whiteHoldM{mode}.jpg', '{theme}/whiteHoldS{mode}.jpg'],
    'key2': ['{theme}/colorNote{mode}.jpg', '{theme}/colorHoldS{mode}.jpg',
             '{theme}/colorHoldM{mode}.jpg', '{theme}/colorHoldS{mode}.jpg'],
}

key_template_map = {
    ord('S'): 'key1', ord('D'): 'key2', ord('F'): 'key1',  # left-hand
    ord('J'): 'key1', ord('K'): 'key2', ord('L'): 'key1',  # right-hand
    ord('V'): 'extra_left', ord('N'): 'extra_right',  # extra red track for 8b l1, r1
    ord('A'): 'side', ord(';'): 'side',  # disk left and right (joystick down)
    # ord('Z'): '', ord('.'): '',  # fever, no detection needed, simply press regularly
}

key_configs = {
    '4B': [
        ['S', 'D', 'K', 'L'],
        ['A', ';', ],
    ],
    '5B': [
        ['S', 'D', 'J', 'K', 'L'],
        ['A', ';'],
    ],
    '6B': [
        ['S', 'D', 'F', 'J', 'K', 'L'],
        ['A', ';'],
    ],
}

key_configs['4X'] = key_configs['4B'] + [['V', 'N']]

key_configs['8B'] = key_configs['6B'] + [['V', 'N']]

BoardLocation = Literal['left', 'middle', 'right']

location_coords: dict[BoardLocation, Tuple[float, float, float, float]] = {
    'left': (120.0, 0.0, 600.0, 785.0),
    'middle': (0.0, 0.0, 0.0, 0.0),
    'right': (0.0, 0.0, 0.0, 0.0),
}


@dataclass
class TrackConfig:
    key: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    button_template: str
    hold_start_template: str
    hold_middle_template: str
    hold_end_template: str


class ModeConfig:
    def __init__(self, theme: str, mode, location):
        self.theme = theme
        self.mode = mode
        self.location = location
        self.coords = location_coords[location]

        self.track_configs = self._init_track_configs(theme, mode, self.coords[0], self.coords[2])

    @staticmethod
    def _init_track_configs(theme, mode, start, end):
        track_num = int(mode[0], 10)
        length = end - start
        key_config = key_configs[mode]

        tracks = []
        for keys in key_config:
            track_start = start
            track_len = length / len(keys)
            for i, key in enumerate(keys):
                track_end = min(track_start + track_len, end)
                key_template = templates[key_template_map[key]]
                template_mode = mode if mode.startswith('4') or mode.startswith('5') else ''
                tracks.append(TrackConfig(
                    key=ord(key),
                    start_x=track_start,
                    end_x=track_end,
                    start_y=start_y,
                    end_y=end_y,
                    button_template=key_template[0].format(theme=theme, mode=template_mode),
                    hold_start_template=key_template[1].format(theme=theme, mode=template_mode),
                    hold_middle_template=key_template[2].format(theme=theme, mode=template_mode),
                    hold_end_template=key_template[3].format(theme=theme, mode=template_mode),
                ))

        return tracks
