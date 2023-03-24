import cv2 as cv
from vmacro.logger import logger

class VideoCapture:
    def __init__(self, video_path: str):
        self._video = cv.VideoCapture(video_path)

    def read(self):
        ret, frame = self._video.read()
        if not ret:
            logger.info("Video finished")
            exit(0)
        return frame

    def __del__(self):
        self._video.release()
