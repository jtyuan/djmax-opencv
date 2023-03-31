from collections import Counter

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


def estimate_offset(detections, predictions, iou_threshold, *, logger):

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

    if len(matches) > len(detections) / 2:
        # at lease half of detections are matched
        matches = np.array(matches, dtype=int)
        offsets = matches[:, 1] - matches[:, 0]
        if not np.all(offsets == offsets[0]):
            # offsets mismatch, rematch based on the most frequent offset
            counter = Counter(list(offsets))
            new_offset = counter.most_common(1)[0][0]
            logger.debug(f"new offset from iou matches: {new_offset}")
            return new_offset
        return offsets[0]

    # otherwise, estimate the offset based on simple assumptions:
    #   1. new detections always above predicted notes
    #   2. old detections (on the bottom, or right in the chart) are always covered in prediction
    # +------detections---------+----offset------+
    # |                                          |
    # |----new dets----+-------predictions-------+
    # offset = len(predictions) - len(detections) + len(new detections)
    h = predictions[-1, 3] - predictions[-1, 1]
    pred_top = predictions[-1, 1]
    new_dets = detections[detections[:, 3] < pred_top - h]
    new_offset = len(predictions) - len(detections) + len(new_dets)
    logger.debug(f"new offset from experience: {new_offset}")
    return new_offset

# def associate(detections, predictions, iou_threshold, *, logger):
#     if len(predictions) == 0:
#         return (
#             np.empty((0, 2), dtype=np.ushort),
#             np.arange(len(detections)),
#             np.empty((0, 5), dtype=np.ushort),
#         )
#
#
#     logger.debug(f"associate predictions: {predictions}")
#     if rematch_needed:
#
#         if new_offset is not None:
#             overlap_size = min(len(predictions) - new_offset, len(detections))
#             if overlap_size > 0:
#                 det_indices = np.arange(0, overlap_size, dtype=np.ushort)
#                 pred_indices = det_indices + new_offset
#                 matches = np.column_stack((det_indices, pred_indices))
#                 matched_detections = set(det_indices)
#                 matched_predictions = set(pred_indices)
#             else:
#                 matches = np.empty((0, 2))
#                 matched_detections = set()
#                 matched_predictions = set()
#
#     for d, det in enumerate(detections):
#         if d not in matched_detections:
#             unmatched_detections.append(d)
#
#     for t, trk in enumerate(predictions):
#         if t not in matched_predictions:
#             unmatched_predictions.append(t)
#
#     logger.debug(f"final matches: {matches}; unmatched_dets: {unmatched_detections}; "
#                  f"unmatched_preds: {unmatched_predictions}")
#     return matches, np.array(unmatched_detections, dtype=np.ushort), np.array(unmatched_predictions, dtype=np.ushort)


def distance(dets, preds,  logger):
    n = min(len(dets), len(preds))
    mean_y_dist = np.mean(np.abs(dets[:n, 3] - preds[:n, 3]))
    if np.isnan(mean_y_dist):
        logger.error(f"mean y dist nan: n={n}, len(dets)={len(dets)}, len(preds)={len(preds)}")
        raise ValueError("Fuck")
    if n > 1:
        mean_gap_diff = np.mean(np.abs((dets[:n - 1, 1] - dets[1:n, 3]) - (preds[:n - 1, 1] - preds[1:n, 3])))
    else:
        mean_gap_diff = 0
    if np.isnan(mean_gap_diff):
        logger.error(f"mean gap diff nan: n={n}, len(dets)={len(dets)}, len(preds)={len(preds)}")
        raise ValueError("Fuck")
    return mean_y_dist + mean_gap_diff


def associate_v2(detections, predictions, distance_threshold, *, logger):
    if len(predictions) == 0:
        return (
            np.empty((0, 2), dtype=np.ushort),
            np.arange(len(detections)),
            np.empty(0, dtype=np.ushort),
        )
    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=np.ushort),
            np.empty(0, dtype=np.ushort),
            np.arange(len(predictions)),
        )

    logger.debug(f"associate predictions: {predictions}")

    m, n = len(detections), len(predictions)
    min_dist = float('inf')
    best_offset = 0
    for i in range(n + m - 1):
        offset = i - (m - 1)
        if offset < 0:  # detected but not tracked (rare)
            dist = distance(detections[-offset:], predictions[0:], logger)
        else:
            dist = distance(detections[0:], predictions[offset:], logger)
        if dist < min_dist:
            min_dist = dist
            best_offset = offset

    if min_dist > distance_threshold:
        logger.debug(f"associate: min distance {min_dist} > {distance_threshold}. Fallback to IoU.")
        best_offset = estimate_offset(detections, predictions, 0.22, logger=logger)

    logger.debug(f"associate: min distance {min_dist}, best offset {best_offset}")
    if best_offset < 0:
        overlap_num = min(m + best_offset, n)
        pred_indices = np.arange(0, overlap_num, dtype=np.ushort)
        det_indices = pred_indices - best_offset
    else:
        overlap_num = min(m, n - best_offset)
        det_indices = np.arange(0, overlap_num, dtype=np.ushort)
        pred_indices = det_indices + best_offset
    matches = np.column_stack((det_indices, pred_indices))
    matched_detections = set(det_indices)
    matched_predictions = set(pred_indices)

    unmatched_detections = []
    unmatched_predictions = []

    for d, det in enumerate(detections):
        if d not in matched_detections:
            unmatched_detections.append(d)

    for t, trk in enumerate(predictions):
        if t not in matched_predictions:
            unmatched_predictions.append(t)

    logger.debug(f"final matches: {matches}; unmatched_dets: {unmatched_detections}; "
                 f"unmatched_preds: {unmatched_predictions}")
    return matches, np.array(unmatched_detections, dtype=np.ushort), np.array(unmatched_predictions, dtype=np.ushort)


if __name__ == '__main__':
    # Test cases
    from loguru import logger as _logger

    dets = np.array([[80, 190, 159, 211, 0.91871, 2]])

    preds = np.array([[80, 766.06, 159, 788.06, 0.801, 2, 0, 4.0982e+05, 706.83, 0],
                      [80, 190.06, 159, 211.06, 0.90213, 2, 1, 4.0982e+05, 706.83, 1]])
    print(associate_v2(dets, preds, 5, logger=_logger))

    # dets = np.array([
    #     [480, 611, 559, 632, 0.92417, 2],
    #     [480, 563, 559, 584, 0.91245, 2],
    #     [480, 251, 559, 272, 0.90828, 2],
    #     [480, 203, 559, 224, 0.91417, 2],
    # ])
    # preds = np.array([
    #     [480, 612.69, 559, 633.69, 0.9312, 2, 80, 3.3836e+05, 248.33, 0],
    #     [480, 565.69, 559, 585.69, 0.91976, 2, 81, 3.3836e+05, 248.33, 1],
    #     [480, 253.69, 559, 273.69, 0.91523, 2, 83, 3.3836e+05, 248.33, 2],
    #     [480, 204.69, 559, 225.69, 0.90593, 2, 84, 3.3836e+05, 248.33, 3],
    # ])
    # print(associate_v2(dets, preds, 5, logger=_logger))
    #
    # dets = np.array([
    #     [439, 661, 560, 685, 0.85351, 1],
    #     [441, 327, 560, 351, 0.86538, 0],
    # ])
    # preds = np.array([
    #     [439, 696.91, 560, 720.91, 0.85351, 1, 33, 3.3905e+05, 652.36, 0],
    #     [441, 362.91, 560, 386.91, 0.86538, 0, 34, 3.3905e+05, 652.36, 1],
    # ])
    # print(associate_v2(dets, preds, 5, logger=_logger))
    #
    # dets = np.array([[441, 403, 560, 428, 0.85722, 0]])
    # preds = np.array([
    #     [441, 403, 560, 428, 0.85722, 0, 35, 3.3905e+05, 652.36],
    #     [439, 661, 560, 685, 0.85351, 1, 33, 3.3905e+05, 652.36],
    #     [439, 661, 560, 685, 0.85351, 0, 34, 3.3905e+05, 652.36],
    # ])
    # print(associate_v2(dets, preds, 5, logger=_logger))
