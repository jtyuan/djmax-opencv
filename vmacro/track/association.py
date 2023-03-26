import numpy as np


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    return o


def iou_batch_y(bboxes1, bboxes2):
    """
    From SORT: Computes 1d-IOU of the y-axis between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    overlap = np.maximum(0.0, yy2 - yy1)
    o = overlap / (
        (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 3] - bboxes2[..., 1])
        - overlap
    )
    return o


def associate(detections, predictions, iou_threshold, *, logger):
    if len(predictions) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    logger.debug(f"associate predictions: {predictions}")
    iou_matrix: np.ndarray = iou_batch_y(detections, predictions)

    matched_indices = None
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)

    if matched_indices is None:
        matched_indices = np.empty(shape=(0, 2))

    logger.debug(f"iou results: {iou_matrix}; indices: {matched_indices}")

    matches = []
    unmatched_detections = []
    unmatched_predictions = []

    # filter matched indices to ensure at-most 1-to-1 matching and remove low iou_threshold results
    matched_detections = set()
    matched_predictions = set()
    for m in matched_indices:
        if (
            iou_matrix[m[0], m[1]] >= iou_threshold
            and m[0] not in matched_detections
            and m[1] not in matched_predictions
        ):
            matches.append(m)
            matched_detections.add(m[0])
            matched_predictions.add(m[1])

    new_offset = None
    if len(matches) > len(detections) / 2:
        # at lease half of detections are matched
        matches = np.array(matches, dtype=int)
        offsets = matches[:, 1] - matches[:, 0]
        if np.all(offsets == offsets[0]):
            # offsets mismatch, rematch based on the most frequent offset
            new_offset = np.argmax(np.bincount(offsets))
            logger.debug(f"new offset from iou matches: {new_offset}")

    if new_offset is None:
        # otherwise, estimate the offset based on simple assumptions:
        #   1. new detections always above predicted notes
        #   2. old detections (on the bottom, or right in the chart) are always covered in prediction
        # +------detections---------+----offset------+
        # |                                          |
        # |----new dets----+-------predictions-------+
        # offset = len(predictions) - len(detections) + len(new detections)
        pred_top = predictions[-1, 1]
        new_dets = detections[detections[:, 3] < pred_top]
        new_offset = len(predictions) - len(detections) + len(new_dets)
        logger.debug(f"new offset from experience: {new_offset}")

    if new_offset is not None:
        overlap_size = min(len(predictions) - new_offset, len(detections))
        if overlap_size > 0:
            det_indices = np.arange(0, overlap_size, dtype=int)
            pred_indices = det_indices + new_offset
            matches = np.column_stack((det_indices, pred_indices))
            matched_detections = set(det_indices)
            matched_predictions = set(pred_indices)
        else:
            matches = np.empty((0, 2))
            matched_detections = set()
            matched_predictions = set()

    for d, det in enumerate(detections):
        if d not in matched_detections:
            unmatched_detections.append(d)

    for t, trk in enumerate(predictions):
        if t not in matched_predictions:
            unmatched_predictions.append(t)

    logger.debug(f"final matches: {matches}; unmatched_dets: {unmatched_detections}; "
                 f"unmatched_preds: {unmatched_predictions}")
    return matches, np.array(unmatched_detections, dtype=int), np.array(unmatched_predictions, dtype=int)


if __name__ == '__main__':
    dets = np.array([[319, 261, 440, 289, 0.889, 2], [319, 64, 440, 91, 0.87867, 2]])
    preds = np.array([[319, 258.81, 440, 286.81, 0.91107, 2, 0, 2.4248e+05, 1937.4],
                      [319, 62.814, 441, 87.814, 0.90449, 2, 1, 2.4248e+05, 1937.4]])
    from loguru import logger
    print(associate(detections=dets, predictions=preds, iou_threshold=0.3, logger=logger))
