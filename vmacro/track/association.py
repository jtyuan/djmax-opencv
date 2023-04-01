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

    matched_detections = set()
    matched_predictions = set()
    if matched_indices.any():
        # filter matched indices to ensure at-most 1-to-1 matching and remove low iou_threshold results
        for m in matched_indices:
            if (
                iou_matrix[m[0], m[1]] >= iou_threshold
                and m[0] not in matched_detections
                and m[1] not in matched_predictions
            ):
                matches.append(m)
                matched_detections.add(m[0])
                matched_predictions.add(m[1])

    else:
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

        if new_offset is not None:
            overlap_size = min(len(predictions) - new_offset, len(detections))
            if overlap_size > 0:
                det_indices = np.arange(0, overlap_size, dtype=np.ushort)
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
    return matches, np.array(unmatched_detections, dtype=np.ushort), np.array(unmatched_predictions, dtype=np.ushort)


def distance(dets, preds, logger):
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


def associate_v2(detections, predictions, *, logger):
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


# def estimate_distance(bboxes1, bboxes2):
#     min_y1_distance = float('inf')
#     min_y2_distance = float('inf')
#
#     for i, bbox1 in enumerate(bboxes1):
#         for j, bbox2 in enumerate(bboxes2):
#             y1_dist = abs(bbox1[1] - bbox2[1])
#             y2_dist = abs(bbox1[3] - bbox2[3])
#
#             if y1_dist < min_y1_distance:
#                 min_y1_distance = y1_dist
#             if y2_dist < min_y2_distance:
#                 min_y2_distance = y2_dist
#
#     if min_y1_distance > min_y2_distance:
#         return 1, min_y1_distance
#     else:
#         return 3, min_y2_distance


def associate_v3(detections, predictions, *, dist_threshold, logger):
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

    matches = []
    unmatched_detections = []
    unmatched_predictions = []

    i, j = 0, 0
    m, n = len(detections), len(predictions)
    matched_predictions = set()
    while i < m and j < n:
        min_dist = float('inf')
        best_j = -1
        for q in range(j, n):
            dist = np.min(np.abs(detections[i, [1, 3]] - predictions[q, [1, 3]]))
            if dist < min_dist:
                best_j = q
                min_dist = dist

        if min_dist <= dist_threshold:
            matches.append((i, best_j))
            i += 1
            j = best_j  # + 1  # allow one i to match multiple j for detection error
            matched_predictions.add(j)
        elif detections[i, 3] > predictions[j, 3]:
            unmatched_detections.append(i)
            i += 1
        else:
            j += 1

    # Add any remaining unmatched indices from both lists
    # logger.debug(f"{j}, {last_j_matched}, {n}, {list(range(1, 1))}")
    unmatched_detections.extend(range(i, m))
    unmatched_predictions = list(sorted(set(range(0, n)) - matched_predictions))

    logger.debug(f"final matches: {matches}; unmatched_dets: {unmatched_detections}; "
                 f"unmatched_preds: {unmatched_predictions}")
    return (
        np.array(matches, dtype=np.ushort),
        np.array(unmatched_detections, dtype=np.ushort),
        np.array(unmatched_predictions, dtype=np.ushort),
    )


if __name__ == '__main__':
    # Test cases
    from loguru import logger as _logger
#
#
#     dets = np.array([[78, 489, 200, 514, 0.92003, 2],
# [79, 57, 199, 81, 0.90864, 2]])
#
#     preds = np.array([[78, 493.49, 200, 518.49, 0.90916, 2, 0, 7.6857e+05, 1009.5, 0],
# [79, 61.49, 199, 86.49, 0.91064, 2, 1, 7.6857e+05, 1009.5, 1]])
#     _logger.debug(associate_v3(dets, preds, dist_threshold=20, logger=_logger))

    # dets = np.array([[319, 323, 559, 357, 0.80083, 4], [320, 24, 560, 45, 0.73777, 3]])
    # preds = np.array([[320, 325.49, 559, 355.49, 0.80145, 4, 0, 7.7479e+05, 838.96, 0]])
    # _logger.debug(associate_v3(dets, preds, dist_threshold=20, logger=_logger))

    dets = np.array([[367, 462, 465, 486, 0.91449, 2]
, [367, 294, 465, 318, 0.90867, 2]
, [367, 126, 464, 150, 0.93253, 2]])
    preds = np.array([[369, 625.17, 465, 649.17, 0.92434, 2, 233, 7.7672e+05, 1023.4, 0]
, [368, 457.17, 465, 481.17, 0.91135, 2, 234, 7.7672e+05, 1023.4, 1]
, [367, 289.17, 464, 314.17, 0.93293, 2, 235, 7.7672e+05, 1023.4, 2]
, [367, 121.17, 464, 145.17, 0.91706, 2, 236, 7.7672e+05, 1023.4, 3]])
    _logger.debug(associate_v3(dets, preds, dist_threshold=20, logger=_logger))
    #
    # dets = np.array([
    #     [480, 611, 559, 632, 0.92417, 2],
    #     [480, 563, 559, 584, 0.91245, 2],
    #     [480, 563, 559, 584, 0.91245, 2],
    #     [480, 251, 559, 272, 0.90828, 2],
    #     [480, 203, 559, 224, 0.91417, 2],
    #     [480, 203, 559, 224, 0.91417, 2],
    # ])
    # dets_iou = np.triu(iou_batch_y(dets, dets), 1)
    # print(dets_iou)
    # if min(dets_iou.shape) > 0:
    #     a = (dets_iou > 0.3).astype(np.int32)
    #     duplicate_indices = np.stack(np.where(a), axis=1)
    #     print(duplicate_indices)
    # preds = np.array([
    #     [480, 612.69, 559, 633.69, 0.9312, 2, 80, 3.3836e+05, 248.33, 0],
    #     [480, 565.69, 559, 585.69, 0.91976, 2, 81, 3.3836e+05, 248.33, 1],
    #     [480, 253.69, 559, 273.69, 0.91523, 2, 83, 3.3836e+05, 248.33, 2],
    #     [480, 204.69, 559, 225.69, 0.90593, 2, 84, 3.3836e+05, 248.33, 3],
    # ])
    # print(associate_v3(dets, preds, dist_threshold=20, logger=_logger))
    #
    # dets = np.array([
    #     [439, 661, 560, 685, 0.85351, 1],
    #     [441, 327, 560, 351, 0.86538, 0],
    # ])
    # preds = np.array([
    #     [439, 696.91, 560, 720.91, 0.85351, 1, 33, 3.3905e+05, 652.36, 0],
    #     [441, 362.91, 560, 386.91, 0.86538, 0, 34, 3.3905e+05, 652.36, 1],
    # ])
    # print(associate_v3(dets, preds, dist_threshold=20, logger=_logger))
    #
    # dets = np.array([[441, 403, 560, 428, 0.85722, 0]])
    # preds = np.array([
    #     [441, 403, 560, 428, 0.85722, 0, 35, 3.3905e+05, 652.36],
    #     [439, 661, 560, 685, 0.85351, 1, 33, 3.3905e+05, 652.36],
    #     [439, 661, 560, 685, 0.85351, 0, 34, 3.3905e+05, 652.36],
    # ])
    # print(associate_v3(dets, preds, dist_threshold=20, logger=_logger))
