# import numpy as np
# from loguru import logger
#
# from vmacro.track.association import iou_batch_y
#
# if __name__ == '__main__':
#     dets = np.array([[81, 549, 320, 582, 0.61965, 7],
#  [81, 547, 322, 582, 0.63599, 6],
#  [82, 546, 320, 579, 0.52375, 5]])
#     detections = dets[dets[:, 3].argsort()][::-1]  # sorted by bottom y
#     logger.debug(f"input detections: {detections}")
#
#     # Find and merge duplicated detections
#     dets_iou = np.triu(iou_batch_y(detections, detections), 1)
#     a = (dets_iou > 0.3).astype(np.int32)
#     print(a)
#     overlapped_indices = np.stack(np.where(a), axis=1)
#     class_ids = [[] for _ in range(len(detections))]
#     rows_to_del = set()
#     print(overlapped_indices)
#     for i, j in overlapped_indices:
#         # Record ids of duplicated rows
#         print(i, j)
#         if i not in rows_to_del:
#             print('in', i, j)
#             class_ids[i].append(detections[j, 5])
#             rows_to_del.add(j)
#     # Remove duplication
#     detections = np.delete(detections, list(rows_to_del), axis=0)
#     for i, det in enumerate(detections):
#         # Record ids of unique rows
#         class_ids[i].append(det[5])