import cv2 as cv

class VideoCapture:
    def __init__(self, video_path: str):
        self._video = cv.VideoCapture(video_path)

    def read(self):
        ret, frame = self._video.read()
        if not ret:
            exit(0)
        return frame

    def __del__(self):
        self._video.release()
