import cv2
import dlib

from src.video_read import Video_Read
from src.utils import image_paint
import psutil
import time

st = time.time()


def main():
    video_read = Video_Read(0)

    model_path = "mmod_human_face_detector.dat"
    model = dlib.cnn_face_detection_model_v1(model_path)

    for frame in video_read:
        faces = model(frame, 1)
        confidence_scores = [face.confidence for face in faces]
        faces = [(face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()) for face in faces]

        for box, confidence_score in zip(faces, confidence_scores):
            image_paint(frame, box, f"{confidence_score:.2f}")

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