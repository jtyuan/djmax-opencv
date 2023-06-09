from dataclasses import dataclass
from typing import Literal

import numpy as np

NoteClass = Literal[
    'hold-end', 'hold-start', 'note',
    'side-end', 'side-start',
    'tbend', 'tbnote', 'tbstart',
    'xend', 'xnote', 'xstart',
]


@dataclass
class Note:
    bbox: np.ndarray
    cls: NoteClass
    timestamp: float
    speed: float
    id: int = -1

    def __eq__(self, other: 'Note'):
        return self.id == other.id
