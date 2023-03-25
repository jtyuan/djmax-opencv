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


def associate(detections, predictions, iou_threshold):
    if len(predictions) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix: np.ndarray = iou_batch_y(detections, predictions)

    matched_indices = None
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)

    if matched_indices is None:
        matched_indices = np.empty(shape=(0, 2))

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
        matches = np.array(matches)
        offsets = matches[:, 1] - matches[:, 0]
        if not np.all(offsets == offsets[0]):
            # offsets mismatch, rematch based on the most frequent offset
            new_offset = np.argmax(np.bincount(offsets))
    else:
        # otherwise, assume new offset to be 0
        new_offset = 0

    if new_offset is not None:
        det_indices = np.arange(0, min(detections.shape[0], predictions.shape[0]))
        pred_indices = det_indices + new_offset
        matches = np.column_stack((det_indices, pred_indices))
        matched_detections = set(det_indices)
        matched_predictions = set(pred_indices)

    for d, det in enumerate(detections):
        if d not in matched_detections:
            unmatched_detections.append(d)

    for t, trk in enumerate(predictions):
        if t not in matched_predictions:
            unmatched_predictions.append(t)

    return matches, np.array(unmatched_detections), np.array(unmatched_predictions)
