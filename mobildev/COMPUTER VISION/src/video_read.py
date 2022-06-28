import cv2


class Video_Read:
    def __init__(self, video_location):
        self.cap = cv2.VideoCapture(video_location)

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    def __iter__(self):
        return self

