from logging import Logger

import cv2
import numpy as np

from vmacro.config import KeyType, frame_bbox
from vmacro.note import NoteClass

hsv_boundary = {
    'side': (
        np.array([75, 100, 100], dtype=np.uint8),
        np.array([110, 255, 255], dtype=np.uint8),
    ),
    'x': (
        np.array([160, 100, 100], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8),
    ),
    'x2': (
        np.array([140, 100, 100], dtype=np.uint8),
        np.array([160, 255, 255], dtype=np.uint8),
    ),
}

crop_offset = 0.1


class DetCleaner:

    def __init__(self, class_names: dict[int, NoteClass]):
        class_name_to_id: dict[NoteClass, int] = {
            v: k for k, v in class_names.items()
        }
        self._end_id_map: dict[KeyType, int] = {
            'normal': class_name_to_id['hold-end'],
            'side': class_name_to_id['side-end'],
            'x': class_name_to_id['xend'],
            # 'x2': class_name_to_id['x2end'],
        }
        self._end_ids = set(self._end_id_map.values())
        self._start_id_map: dict[KeyType, int] = {
            'normal': class_name_to_id['hold-start'],
            'side': class_name_to_id['side-start'],
            'x': class_name_to_id['xstart'],
            # 'x2': class_name_to_id['x2start'],
        }
        self._start_ids = set(self._start_id_map.values())
        self._normal_id_map: dict[KeyType, int] = {
            'normal': class_name_to_id['note'],
            'x': class_name_to_id['xnote'],
            # 'x2': class_name_to_id['x2note'],
        }
        self._normal_ids = set(self._normal_id_map.values())

    def clean(self, key_type: KeyType, dets: np.ndarray, im0, logger: Logger) -> np.ndarray:
        """Attempt to clean mislabeled hold start/end or normal note classes."""
        if len(dets) == 0:
            return dets
        # run basic calculation on the first det only
        # assumption: notes on the same track have similar dimensions
        if key_type in hsv_boundary:
            boundary = hsv_boundary[key_type]
        else:
            h = dets[0, 3] - dets[0, 1]
            w = dets[0, 2] - dets[0, 0]
            y_offset = h * crop_offset
            x0_offset = w * crop_offset
            x1_offset = w * (0.5 + crop_offset)
            crop = im0[
                   int(dets[0, 1] + y_offset): int(dets[0, 3] - y_offset),
                   int(dets[0, 0] + x0_offset): int(dets[0, 2] - x1_offset)
                   ]
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mean = cv2.mean(hsv_crop)
            lower_bound = np.array(
                [max(mean[0] - 10, 0), max(mean[1] - 50, 0), max(mean[2] - 50, 0)],
                dtype=np.uint8
            )
            upper_bound = np.array(
                [min(mean[0] + 10, 180), min(mean[1] + 50, 255), min(mean[2] + 50, 255)],
                dtype=np.uint8
            )
            boundary = (lower_bound, upper_bound)

        h = dets[:, 3] - dets[:, 1]
        w = dets[:, 2] - dets[:, 0]
        y_offset = (h * crop_offset).astype(np.ushort)
        x0_offset = (w * crop_offset).astype(np.ushort)
        x1_offset = (w * (0.5 + crop_offset)).astype(np.ushort)
        bboxes_above = dets[:, :4].astype(np.ushort)
        bboxes_above[:, 0] += x0_offset
        bboxes_above[:, 1] = np.clip(bboxes_above[:, 1] - 2 * y_offset, frame_bbox[1], frame_bbox[3])
        bboxes_above[:, 2] -= x1_offset
        bboxes_above[:, 3] = np.clip(bboxes_above[:, 1] - y_offset, frame_bbox[1], frame_bbox[3])
        bboxes_below = dets[:, :4].astype(np.ushort)
        bboxes_below[:, 0] += x0_offset
        bboxes_below[:, 1] = np.clip(bboxes_below[:, 3] + y_offset, frame_bbox[1], frame_bbox[3])
        bboxes_below[:, 2] -= x1_offset
        bboxes_below[:, 3] = np.clip(bboxes_below[:, 3] + 2 * y_offset, frame_bbox[1], frame_bbox[3])

        for i, det in enumerate(dets):
            # det: [x1, y1, x2, y2, conf, class_id]
            if det[5] not in self._normal_ids:
                continue

            new_cls = None
            if bboxes_below[i][1] < bboxes_below[i][3] and bboxes_below[i][0] < bboxes_below[i][2]:
                crop_below = im0[bboxes_below[i][1]:bboxes_below[i][3], bboxes_below[i][0]:bboxes_below[i][2]]
                if self._color_in_range(crop_below, boundary[0], boundary[1]):
                    new_cls = self._end_id_map[key_type]

            if bboxes_above[i][1] < bboxes_above[i][3] and bboxes_above[i][0] < bboxes_above[i][2]:
                crop_above = im0[bboxes_above[i][1]:bboxes_above[i][3], bboxes_above[i][0]:bboxes_above[i][2]]
                if self._color_in_range(crop_above, boundary[0], boundary[1]):
                    new_cls = self._start_id_map[key_type]

            if new_cls:
                logger.debug(f"Update detection class: {det[5]} -> {new_cls} (det: {det})")
                det[5] = new_cls

        return dets

    def get_key_type(self, bbox, im0) -> KeyType:
        """Check cropped area in the given box to see if a note is an x or x2 note."""
        from loguru import logger
        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]
        y_offset = h * crop_offset
        x0_offset = w * crop_offset
        x1_offset = w * (0.5 + crop_offset)
        crop = im0[
               int(bbox[1] + y_offset): int(bbox[3] - y_offset),
               int(bbox[0] + x0_offset): int(bbox[2] - x1_offset)
               ]
        to_check: list[KeyType] = ['x2', 'x', 'side']

        hsv_image = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        logger.info(f"Bbox crop mean hsv: {cv2.mean(hsv_image)}")
        for key_type in to_check:
            logger.info(f"Checking bbox {list(bbox)} against {list(hsv_boundary[key_type])}")
            if self._color_in_range(crop, hsv_boundary[key_type][0], hsv_boundary[key_type][1]):
                logger.info(f"Bbox {list(bbox)} found in {key_type}")
                return key_type
            logger.info(f"Bbox {list(bbox)} not in {key_type}")
        return 'normal'

    @staticmethod
    def _color_in_range(image, lower_bound, upper_bound, threshold=0.5):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        num_pixels_in_range = np.sum(mask == 255)
        total_pixels = len(mask) * mask.shape[1]
        return num_pixels_in_range / total_pixels > threshold
