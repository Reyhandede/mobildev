import cv2
from facenet_pytorch import MTCNN

from src.video_read import Video_Read
from src.utils import image_paint
import psutil
import time

st = time.time()


def main():
    video_read = Video_Read(0)

    model = MTCNN(image_size=300, margin=0, min_face_size=40,
                  thresholds=[0.6, 0.7, 0.7], factor=0.709,
                  post_process=True)

    for frame in video_read:
        faces, possibilities, all_face_points = model.detect(frame, landmarks=True)
        if faces is not None:
            for box, possibiliti, face_points in zip(faces, possibilities, all_face_points):
                image_paint(frame, box, f"{possibiliti:.2f}", face_points)

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