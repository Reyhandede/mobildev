import cv2

from src.video_read import Video_Read
from src.utils import image_paint
import psutil
import time

st = time.time()


def main():
    video_read = Video_Read(0)

    model_path = "haarcascade_frontalface_alt.xml"
    model = cv2.CascadeClassifier(model_path)

    for frame in video_read:
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        gray_scale = cv2.equalizeHist(gray_scale)
        faces = model.detectMultiScale(gray_scale) 
        faces = [(xmin, ymin, xmin+width, ymin+height) for xmin, ymin, width, height in faces]

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