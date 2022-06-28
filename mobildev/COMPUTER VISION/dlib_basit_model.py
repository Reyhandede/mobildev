import cv2
import dlib
from src.video_read import Video_Read
from src.utils import image_paint
import os
import psutil
import time

st = time.time()


def main():
    video_read = Video_Read(0)

    model = dlib.get_frontal_face_detector()

    for frame in video_read:
        faces = model(frame, 1)
        faces = [(face.left(), face.top(), face.right(), face.bottom()) for face in faces]

        

        for box in faces:
            image_paint(frame, box)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()


pid = os.getpid()
python_process = psutil.Process(pid)
memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)


et = time.time()
print('Execution time:', et-st, 'seconds')