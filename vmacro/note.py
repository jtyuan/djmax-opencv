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
    id: int
    bbox: np.ndarray
    cls: NoteClass
    timestamp: float
